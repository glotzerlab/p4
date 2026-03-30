import hoomd

import p4
import coxeter
import numpy as np

# cube_vertices = np.array([
#     [-1, -1, -1],
#     [-1, -1,  1],
#     [-1,  1, -1],
#     [-1,  1,  1],
#     [ 1, -1, -1],
#     [ 1, -1,  1],
#     [ 1,  1, -1],
#     [ 1,  1,  1]
# ])
# cube_faces = [
#     [0, 2, 6, 4],
#     [0, 4, 5, 1],
#     [4, 6, 7, 5],
#     [0, 1, 3, 2],
#     [2, 3, 7, 6],
#     [1, 5, 7, 3],
# ]
# concave_vertices = np.array([
#     [-1, -1, -1],
#     [-1,  1, -1],
#     [ 1,  1, -1],
#     [ 1, -1, -1],
#     [-1,  0,  0],
#     [ 1,  0,  0],
#     [-1, -1,  1],
#     [-1,  1,  1],
#     [ 1,  1,  1],
#     [ 1, -1,  1],
# ])
# concave_faces = [
#     [0, 1, 2, 3],
#     [0, 3, 5, 4],
#     [4, 5, 9, 6],
#     [3, 2, 8, 9, 5],
#     [2, 1, 7, 8],
#     [1, 0, 4, 6, 7],
#     [6, 9, 8, 7],
# ]

# analyte = p4.Body(
#     primary_type="A",
#     secondary_types=["B"],
#     positions_by_type=dict(
#         # B=cube_vertices
#         B=concave_vertices
#     ),
# )

# # figure, traces = body.plot(
# #     type_shapes={
# #         # "A": coxeter.shapes.Polyhedron(cube_vertices, cube_faces),
# #         "A": coxeter.shapes.Polyhedron(concave_vertices, concave_faces),
# #         # "B": coxeter.shapes.ConvexPolyhedron([[-0.2, -0.2, -0.2],[-0.2, -0.2,  0.2],[-0.2,  0.2, -0.2],[-0.2,  0.2,  0.2],[ 0.2, -0.2, -0.2],[ 0.2, -0.2,  0.2],[ 0.2,  0.2, -0.2],[ 0.2,  0.2,  0.2]])
# #     },
# #     type_styles=dict(A=dict(opacity=0.3)),
# #     slice={"x": -1},        # right
# #     # slice={"y": -0.5},
# #     # slice={"y": -1},        # wrong
# #     # slice={"y": -0.9999999},        # right
# #     # slice={"y": -1, "x": -1},   # expect o     o   --> wrong
# #     # slice={"y": 0, "x": -1},    # expect ---o---   --> right
# #     # slice={"y": 1, "x": -1},      # expect o-----o   --> wrong
# #     # ignore_types=["B"]
# #     schematic_slice=True,
# #     # schematic_slice_scale=1.25,
# #     schematic_slice_opacity=0.5,
# #     schematic_slice_color="red",
# #     # schematic_slice_line_width=50
# # )

# # figure.show()


# interactions = [
#     # surface sites interact with each other via attractive gaussian
#     p4.Interaction(
#         hoomd_class=hoomd.md.pair.Gaussian,
#         initial_args={},
#         default_params=dict(r_cut=0, params=dict(epsilon=0, sigma=1)),
#         typed_params={
#             ("B", "C"): dict(r_cut=3, params=dict(epsilon=-1, sigma=1))
#         }
#     ),
#     # cores interact with surface sites via weak ALJ
#     p4.Interaction(
#         hoomd_class=hoomd.md.pair.aniso.ALJ,
#         initial_args={},
#         default_params=dict(
#             r_cut=0,
#             shape=dict(faces=[], vertices=[]),
#             params=dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=0)),
#         typed_params={
#             ("A", "C"): dict(r_cut=3, params=dict(epsilon=0.1, sigma_i=0.2, sigma_j=0.2, alpha=0)),
#             "A": dict(shape=dict(vertices=concave_vertices, faces=concave_faces))
#         }
#     )
# ]

# probe = p4.Body("C")

# system = p4.System(probe, analyte, interactions)

# field = p4.Field.from_csv("concave_plotting_test_2d.csv", "mean")
field = p4.Field.from_csv("lj-sphere-uft.csv")
a = field.aggregate_over_orientations("F", "mean")
fig, tr = a.plot(vectors=True)
fig.show()