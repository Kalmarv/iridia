//! The ray tracing pipeline's programs, which a library holds (ray generation, intersection, any-hit, closest-hit, miss,
//! callable): the payload and attributes a program's function takes, which are variables of the IR's ray payload and
//! hit attributes spaces laid out in bytes as a task payload is ([`super::payload`]); the ray's system values; and the
//! pipeline's operations as the IR's (`TraceRay`, `CallShader` and `ReportHit` of the variables the memory they hand
//! over lives in, `IgnoreHit` and `AcceptHitAndEndSearch` as the ends of an any-hit program).

use iridia_ir::{
    Node, Op, Ray, Space, Terminator, Type, Var, VarId,
    interface::{Stage, StageFacts, SystemValue},
};

use super::{Lift, Lv, Pointer};
use crate::Result;
use crate::lift::setup::ray_system_value;
use dxil_reader::module::{Instruction, Type as LlvmType, ValueId};

impl Lift<'_> {
    /// The variables of the payload and attributes a ray tracing program's function takes, its pointer parameters in
    /// order (the payload, or a callable program's parameter, then a hit's attributes): each the words of the bytes the
    /// facts state, laid out in bytes.
    pub(crate) fn ray_parameters(&mut self) -> Result<()> {
        let stage = self.setup.interface.stage;
        if !stage.is_ray_tracing() {
            return Ok(());
        }
        let StageFacts::RayTracing(facts) = self.setup.interface.facts.clone() else {
            return self.refuse(format!("a {stage:?} program without its facts"));
        };
        let spaces: &[(Space, u32, &str)] = match stage {
            Stage::AnyHit | Stage::ClosestHit => &[
                (Space::RayPayload, facts.payload, "payload"),
                (Space::HitAttributes, facts.attributes, "attributes"),
            ],
            Stage::Miss | Stage::Callable => &[(Space::RayPayload, facts.payload, "payload")],
            _ => &[],
        };
        let params = match self
            .m
            .value_types
            .get(self.f.value)
            .and_then(|&t| self.m.types.get(t))
        {
            Some(LlvmType::Pointer(t, _)) => match self.m.types.get(*t) {
                Some(LlvmType::Function { params, .. }) => params.len(),
                _ => 0,
            },
            _ => 0,
        };
        if params != spaces.len() {
            return self.refuse(format!(
                "a {stage:?} program whose function takes {params} parameters where {} are expected",
                spaces.len()
            ));
        }
        for (k, &(space, bytes, name)) in spaces.iter().enumerate() {
            if bytes == 0 {
                return self.refuse(format!("a {stage:?} program's {name} of no bytes"));
            }
            let var = VarId(u32::try_from(self.g.vars.len()).unwrap_or(u32::MAX));
            self.g.vars.push(Var {
                name: name.into(),
                ty: Type::U32,
                len: Some(bytes.div_ceil(4)),
                space,
                init: Vec::new(),
            });
            self.bytes.insert(var);
            let parameter = self.f.first.saturating_add(k);
            self.define(
                parameter,
                Lv::Pointer(Pointer {
                    var,
                    at: 0,
                    index: None,
                }),
            )?;
        }
        Ok(())
    }

    /// The ray tracing `dx.op`s: the ray's system values (141 to 154, 161, 213) and the pipeline's operations (155 to
    /// 159).
    pub(crate) fn ray_op(&mut self, inst: &Instruction, code: u64, args: &[ValueId]) -> Result<()> {
        match code {
            // ObjectToWorld(row, column), WorldToObject(row, column): a lane of a row of the instance's transform
            151 | 152 => {
                let (Some(row), Some(column)) = (self.const_arg(args, 1), self.const_arg(args, 2)) else {
                    return self.refuse("a transform read at a computed row or column");
                };
                let (Ok(row @ 0..=2), Ok(column @ 0..=3)) = (u32::try_from(row), u8::try_from(column)) else {
                    return self.refuse("a transform read outside its 3 by 4");
                };
                let system = if code == 151 {
                    SystemValue::ObjectToWorld
                } else {
                    SystemValue::WorldToObject
                };
                let x = self.system_row(system, row, column)?;
                self.varying_result(inst, x, iridia_ir::Scalar::U32)
            }
            155 | 156 => {
                if self.setup.interface.stage != Stage::AnyHit {
                    return self.refuse("a hit decided outside an any-hit program");
                }
                // the program ends here; what follows in the block (dxc's `unreachable`) is not reached
                self.terminate(if code == 155 {
                    Terminator::IgnoreHit
                } else {
                    Terminator::AcceptHitAndEndSearch
                });
                self.decided = true;
                Ok(())
            }
            // TraceRay(scene, flags, mask, hit group offset, hit group stride, miss index, origin, t min, direction,
            // t max, payload)
            157 => {
                let (scene, _) = self.handle(self.arg(args, 1)?)?;
                let mut words = Vec::with_capacity(5);
                for k in 2..=6 {
                    words.push(self.uint(self.arg(args, k)?)?);
                }
                let origin = self.float3(args, 7)?;
                let t_min = self.float(self.arg(args, 10)?)?;
                let direction = self.float3(args, 11)?;
                let t_max = self.float(self.arg(args, 14)?)?;
                let payload = self.handed(self.arg(args, 15)?, "TraceRay's payload")?;
                let &[flags, mask, hit_group_offset, hit_group_stride, miss_index] = words.as_slice() else {
                    return self.refuse("a ray of other than five words");
                };
                self.g.node(Node::TraceRay {
                    scene,
                    ray: Ray {
                        flags,
                        mask,
                        hit_group_offset,
                        hit_group_stride,
                        miss_index,
                        origin,
                        t_min,
                        direction,
                        t_max,
                    },
                    payload,
                });
                Ok(())
            }
            // ReportHit(t, hit kind, attributes): whether the hit was accepted
            158 => {
                let t = self.float(self.arg(args, 1)?)?;
                let kind = self.uint(self.arg(args, 2)?)?;
                let attributes = self.handed(self.arg(args, 3)?, "ReportHit's attributes")?;
                let accepted = self.g.value(Type::BOOL, Op::ReportHit { t, kind, attributes });
                self.define_result(inst, Lv::Value(accepted))
            }
            // CallShader(index, parameter)
            159 => {
                let index = self.uint(self.arg(args, 1)?)?;
                let parameter = self.handed(self.arg(args, 2)?, "CallShader's parameter")?;
                self.g.node(Node::CallShader { index, parameter });
                Ok(())
            }
            _ => {
                let Some(system) = ray_system_value(code) else {
                    return self.refuse(format!("ray tracing dx.op {code}"));
                };
                // a vector's lane is the call's last operand; a scalar has none
                let component = match args.len() {
                    1 => None,
                    _ => match self.const_arg(args, 1).and_then(|c| u8::try_from(c).ok()) {
                        Some(c @ 0..=2) => Some(c),
                        _ => return self.refuse(format!("{system:?} read at a computed or past lane")),
                    },
                };
                let x = self.system_input(system, component)?;
                self.varying_result(inst, x, iridia_ir::Scalar::U32)
            }
        }
    }

    /// The `vec3<f32>` of three float operands from `first`.
    fn float3(&mut self, args: &[ValueId], first: usize) -> Result<iridia_ir::ValueId> {
        let mut lanes = Vec::with_capacity(3);
        for k in first..first.saturating_add(3) {
            lanes.push(self.float(self.arg(args, k)?)?);
        }
        Ok(self
            .g
            .value(Type::vector(iridia_ir::Scalar::F32, 3), Op::Compose(lanes)))
    }

    /// The variable a ray tracing operation hands over (`what`): a whole variable laid out in bytes.
    fn handed(&mut self, v: ValueId, what: &str) -> Result<VarId> {
        match self.get(v)? {
            Lv::Pointer(Pointer {
                var,
                at: 0,
                index: None,
            }) if self.bytes.contains(&var) => Ok(var),
            _ => self.refuse(format!("{what}, which is not a whole variable")),
        }
    }

    /// Lane `column` of row `row` of a system value that is a matrix (an input varying per row, the label's index the
    /// row), as a `u32`.
    fn system_row(&mut self, system: SystemValue, row: u32, column: u8) -> Result<iridia_ir::ValueId> {
        let Some(slot) = self
            .setup
            .interface
            .inputs
            .iter()
            .position(|v| v.system == Some(system) && v.label.index == row)
        else {
            return self.refuse(format!("row {row} of {system:?} has no input"));
        };
        let scalar = self
            .setup
            .interface
            .inputs
            .get(slot)
            .map(|v| iridia_ir::value_type(v.value).scalar)
            .ok_or_else(|| crate::error(self.at, "an input past the interface"))?;
        let slot = u32::try_from(slot).map_err(|_| crate::error(self.at, "an input slot past 2^32"))?;
        let x = self
            .g
            .value(Type::scalar(scalar), Op::LoadInput(slot, 1 << column));
        self.retype(x, Type::U32)
    }
}
