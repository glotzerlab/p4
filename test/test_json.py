import tempfile

import numpy as np
import pytest

import p4
import hoomd
import json
from pathlib import Path


REFERENCE_FOLDER = Path(__file__).parent / "data"

# TODO: return here - add mass and moi to json tests

def cube_vertices(side_length):
    s = side_length
    return [
        [-s/2, -s/2, -s/2],
        [-s/2, -s/2,  s/2],
        [-s/2,  s/2, -s/2],
        [-s/2,  s/2,  s/2],
        [ s/2, -s/2, -s/2],
        [ s/2, -s/2,  s/2],
        [ s/2,  s/2, -s/2],
        [ s/2,  s/2,  s/2]
    ]

def cube_faces():
    return [
        [0, 2, 6, 4],
        [0, 4, 5, 1],
        [4, 6, 7, 5],
        [0, 1, 3, 2],
        [2, 3, 7, 6],
        [1, 5, 7, 3],
    ]

INTERACTION_KWARGS = dict(
    hoomd_class=hoomd.md.pair.aniso.ALJ,
    initial_args=dict(),
    default_params=dict(
        r_cut=np.float32(0),
        params=dict(
            epsilon=0,
            sigma_i=1,
            sigma_j=1,
            alpha=0
        ),
        shape=dict(
            vertices=[],
            faces=[]
        )
    ),
    typed_params={
        ("A", "C"): dict(
            r_cut=5,
            params=dict(
                epsilon=2,
                sigma_i=2,
                sigma_j=2,
                alpha=np.int64(0)
            )
        ),
        "A": dict(
            shape=dict(
                vertices=np.array(cube_vertices(2)),
                faces=cube_faces()
            )
        ),
        "C": dict(
            shape=dict(
                vertices=cube_vertices(2),
                faces=cube_faces()
            )
        )
    }
)

BODY_KWARGS = dict(
    primary_type="A",
    secondary_types=["B"],
    positions_by_type=dict(B=cube_vertices(2)),
    mass_by_type=dict(B=np.float32(2))
)

ARRANGEMENT_KWARGS = dict(
    bodies=[
        p4.Body(
            primary_type="A",
            secondary_types=["B", "C"],
            positions_by_type=dict(
                B=[np.array([-1,0,0])],
                C=np.array([[1,0,0]])
            )
        ),
        p4.Body(
            primary_type="D",
            secondary_types=["E", "F"],
            positions_by_type=dict(
                E=[[0, -1, 0], [0, 1, 0]],
                F=[[0, 0, -1], [0, 0, 1]]
            ),
            orientations_by_type=dict(
                E=[[1,0,0,0], [0, 0.707, 0.707, 0]],
                F=[[0.707, 0, -0.707, 0], [0.707, 0, 0.707, 0]]
            ),
        )
    ],
    positions_by_type=dict(
        A=[[10, 0, 0]],
        D=[[0, 10, 0], [0, 20, 0]]
    ),
    orientations_by_type=dict(
        A=[[1,0,0,0]],
        D=[[0, 0.707, 0.707, 0], [0.707, 0, 0.707, 0]]
    )
)

SYSTEM_WITH_BODY_KWARGS = dict(
    probe=p4.Body("Y", ["Z"], dict(Z=cube_vertices(2))),
    analyte=p4.Body(**BODY_KWARGS),
    interactions=[p4.Interaction(**INTERACTION_KWARGS)]
)

SYSTEM_WITH_ARRANGEMENT_KWARGS = dict(
    probe=p4.Body("Y", ["Z"], dict(Z=cube_vertices(2))),
    analyte=p4.Arrangement(**ARRANGEMENT_KWARGS),
    interactions=[p4.Interaction(**INTERACTION_KWARGS)]
)

def compare_text_files(file_path_1, file_path_2):
    """Raise an Error if two text files do not have identical contents, ignoring different newlines."""
    with open(file_path_1) as file1, open(file_path_2) as file2:
        file1_contents, file2_contents = file1.readlines(), file2.readlines()
        assert file1_contents == file2_contents

@pytest.mark.parametrize("object_type", ["interaction", "body", "arrangement", "system-with-body", "system-with-arrangement"])
@pytest.mark.parametrize("place", ["root", "path"])
@pytest.mark.parametrize("indent_type", ["indent", "noindent"])
def test_to_json(object_type, place, indent_type):
    """Ensure object export to JSON produces files identical to the control."""
    if object_type == "interaction":
        obj = p4.Interaction(**INTERACTION_KWARGS)
    elif object_type == "body":
        obj = p4.Body(**BODY_KWARGS)
    elif object_type == "arrangement":
        obj = p4.Arrangement(**ARRANGEMENT_KWARGS)
    elif object_type == "system-with-body":
        obj = p4.System(**SYSTEM_WITH_BODY_KWARGS)
    elif object_type == "system-with-arrangement":
        obj = p4.System(**SYSTEM_WITH_ARRANGEMENT_KWARGS)

    ref_path = Path(REFERENCE_FOLDER) / f"{object_type}_at_{place}_{indent_type}.json"
    indent = None if indent_type == "noindent" else 2

    with tempfile.TemporaryDirectory(dir=REFERENCE_FOLDER) as tempdir:
        test_path = Path(tempdir) / f"test_{object_type}_at_{place}_{indent_type}.json"
        
        json_path = "." if place == "root" else f"path.to.{object_type}"
        
        obj.to_json(test_path, json_path=json_path, indent=indent)
        
        compare_text_files(ref_path, test_path)

@pytest.mark.parametrize("object_type", ["interaction", "body", "arrangement", "system-with-body", "system-with-arrangement"])
@pytest.mark.parametrize("place", ["root", "path"])
@pytest.mark.parametrize("indent_type", ["indent", "noindent"])
def test_from_json(object_type, place, indent_type):
    """Ensure object creation from JSON produces the expected result."""
    if object_type == "interaction":
        ref_obj = p4.Interaction(**INTERACTION_KWARGS)
    elif object_type == "body":
        ref_obj = p4.Body(**BODY_KWARGS)
    elif object_type == "arrangement":
        ref_obj = p4.Arrangement(**ARRANGEMENT_KWARGS)
    elif object_type == "system-with-body":
        ref_obj = p4.System(**SYSTEM_WITH_BODY_KWARGS)
    elif object_type == "system-with-arrangement":
        ref_obj = p4.System(**SYSTEM_WITH_ARRANGEMENT_KWARGS)

    ref_path = Path(REFERENCE_FOLDER) / f"{object_type}_at_{place}_{indent_type}.json"
    json_path = "." if place == "root" else f"path.to.{object_type}"

    if object_type == "interaction":
        test_obj = p4.Interaction.from_json(ref_path, json_path)
    elif object_type == "body":
        test_obj = p4.Body.from_json(ref_path, json_path)
    elif object_type == "arrangement":
        test_obj = p4.Arrangement.from_json(ref_path, json_path)
    elif object_type == "system-with-body":
        test_obj = p4.System.from_json(ref_path, json_path)
    elif object_type == "system-with-arrangement":
        test_obj = p4.System.from_json(ref_path, json_path)

    assert ref_obj == test_obj


if __name__ == "__main__":
    # Regenerate control files
    object_types = [
        "interaction",
        "body",
        "arrangement",
        "system-with-body",
        "system-with-arrangement"
    ]
    for object_type in object_types:
        for place in ["root", "path"]:
            for indent_type in ["indent", "noindent"]:
                if object_type == "interaction":
                    obj = p4.Interaction(**INTERACTION_KWARGS)
                elif object_type == "body":
                    obj = p4.Body(**BODY_KWARGS)
                elif object_type == "arrangement":
                    obj = p4.Arrangement(**ARRANGEMENT_KWARGS)
                elif object_type == "system-with-body":
                    obj = p4.System(**SYSTEM_WITH_BODY_KWARGS)
                elif object_type == "system-with-arrangement":
                    obj = p4.System(**SYSTEM_WITH_ARRANGEMENT_KWARGS)

                file_path = (
                    Path(REFERENCE_FOLDER)
                    / f"{object_type}_at_{place}_{indent_type}.json"
                )
                json_path = "." if place == "root" else f"path.to.{object_type}"
                indent = None if indent_type == "noindent" else 2
                
                obj.to_json(file_path, json_path=json_path, indent=indent)
