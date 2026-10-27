//! A mesh program: a thread group that writes one meshlet. What it writes is the module's output variable (see
//! [`iridia::build::MeshOutput`]): the counts, a vertex type with a member for each output varying and the
//! vertices' indices of each primitive.

use iridia::build::{Kind, Val};
use naga::Expression as E;

use super::Lift;
use super::setup::const_int;
use crate::Result;
use dxil_reader::module::{Instruction, ValueId};

impl Lift<'_> {
    /// The output variable and its vertex type, once the entry point is declared.
    pub(super) fn start_mesh(&mut self) -> Result<()> {
        let iridia::StageFacts::Mesh(facts) = self.setup.interface.facts.clone() else {
            return self.refuse("a mesh program without its facts");
        };
        let vertex =
            iridia::mesh_vertex_io(&self.setup.interface).map_err(|detail| crate::error(self.at, detail))?;
        if vertex.len() != self.setup.interface.outputs.len() {
            return self.refuse("clip distances or cull distances of a mesh program are not lifted");
        }
        let output = self
            .b
            .declare_mesh_output(&facts, &vertex)
            .map_err(|detail| crate::error(self.at, detail))?;
        self.notes.insert(
            "a mesh program writes its vertices and primitives into the module's output variable; setting the \
             counts is a store every invocation of the thread group makes (D3D requires them to be group-uniform)"
                .into(),
        );
        self.mesh = Some((output, facts));
        Ok(())
    }

    /// A pointer into the output variable: field `field`, and if given, element `index`.
    fn mesh_field(&mut self, field: u32, index: Option<Val>) -> Result<naga::Handle<E>> {
        let Some((output, _)) = self.mesh.as_ref() else {
            return self.refuse("a mesh output written by a program that is no mesh program");
        };
        let global = output.global;
        let base = self.b.global_ptr(global);
        let part = self.b.app(E::AccessIndex { base, index: field });
        Ok(match index {
            Some(i) => self.b.app(E::Access {
                base: part,
                index: i.expr,
            }),
            None => part,
        })
    }

    /// The mesh `dx.op`s: `SetMeshOutputCounts` (168), `EmitIndices` (169), `GetMeshPayload` (170), `StoreVertexOutput`
    /// (171), `StorePrimitiveOutput` (172) and `DispatchMesh` (173).
    pub(super) fn mesh_op(&mut self, _inst: &Instruction, code: u64, args: &[ValueId]) -> Result<()> {
        let Some((output, facts)) = self.mesh.clone() else {
            return self.refuse("a mesh operation outside a mesh program");
        };
        match code {
            168 => {
                let vertices = self.uint(self.arg(args, 1)?)?;
                let primitives = self.uint(self.arg(args, 2)?)?;
                let p = self.mesh_field(output.vertex_count, None)?;
                self.b.store(p, vertices);
                let p = self.mesh_field(output.primitive_count, None)?;
                self.b.store(p, primitives);
                Ok(())
            }
            169 => {
                let primitive = self.uint(self.arg(args, 1)?)?;
                let mut indices = Vec::new();
                for k in 0..facts.topology.indices() as usize {
                    indices.push(self.uint(self.arg(args, 2usize.saturating_add(k))?)?);
                }
                let value = self.b.compose(&indices);
                let element = self.mesh_field(output.primitives, Some(primitive))?;
                let target = self.b.app(E::AccessIndex {
                    base: element,
                    index: output.indices_member,
                });
                self.b.store(target, value);
                Ok(())
            }
            171 => {
                let (Some(id), Some(row), Some(col)) = (
                    const_int(self.m, self.f, self.arg(args, 1)?),
                    const_int(self.m, self.f, self.arg(args, 2)?),
                    const_int(self.m, self.f, self.arg(args, 3)?),
                ) else {
                    return self.refuse("a vertex output written at a computed element, row or column");
                };
                let Some(element) = self
                    .setup
                    .output_elements
                    .iter()
                    .find(|e| i64::from(e.id) == id)
                    .cloned()
                else {
                    return self.refuse(format!("output element {id}"));
                };
                if !element.placed {
                    return self.refuse(format!("output {} is not placed in a register", element.name));
                }
                let (register, lane) = self.cell_of(element.row, element.column, row, col)?;
                let Some((varying, position)) = self.setup.output_cell(register, lane) else {
                    return self.refuse(format!(
                        "output {} has no lane at register {register}",
                        element.name
                    ));
                };
                let Some(carried) = self.setup.interface.outputs.get(varying).map(|v| v.value) else {
                    return self.refuse(format!(
                        "output varying {varying}, which the interface does not have"
                    ));
                };
                let value = self.scalar(self.arg(args, 4)?)?;
                let value = self.b.bitcast_to(value, Kind::of(carried.scalar));
                let vertex = self.uint(self.arg(args, 5)?)?;
                let element = self.mesh_field(output.vertices, Some(vertex))?;
                let member = self.b.app(E::AccessIndex {
                    base: element,
                    index: u32::try_from(varying)
                        .map_err(|_| crate::error(self.at, "more than 4294967295 outputs"))?,
                });
                let target = if carried.width > 1 {
                    self.b.app(E::AccessIndex {
                        base: member,
                        index: u32::from(position),
                    })
                } else {
                    member
                };
                self.b.store(target, value);
                Ok(())
            }
            170 => self
                .refuse("a mesh program that reads a task payload (amplification programs are not lifted)"),
            172 => self.refuse("per-primitive outputs of a mesh program are not lifted"),
            173 => self.refuse("DispatchMesh (amplification programs are not lifted)"),
            other => self.refuse(format!("mesh operation {other}")),
        }
    }
}
