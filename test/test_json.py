import tempfile

import pytest

import p4
import hoomd
import json
from pathlib import Path


REFERENCE_FOLDER = Path(__file__).parent / "data"


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

def make_body_for_json():
    """Return the body used for the reference body JSON files."""
    return p4.Body(
        primary_type="A",
        secondary_types=["B"],
        positions_by_type=dict(B=cube_vertices(2))
    )

def make_interaction_for_json():
    """Return the interaction used for the reference interaction JSON files."""
    return p4.Interaction(
        hoomd_class=hoomd.md.pair.aniso.ALJ,
        initial_args=dict(),
        default_params=dict(
            r_cut=0,
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
                    alpha=0
                )
            ),
            "A": dict(
                shape=dict(
                    vertices=cube_vertices(2),
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

def make_system_for_json():
    """Return the system used for the reference system JSON files."""
    return p4.System(
        probe=make_body_for_json(),
        analyte=p4.Body("C", ["D"], dict(D=cube_vertices(2))),
        interactions=[make_interaction_for_json()]
    )

def compare_text_files(file_path_1, file_path_2):
    """Raise an Error if two text files do not have identical contents, ignoring different newlines."""
    with open(file_path_1) as file1, open(file_path_2) as file2:
        file1_contents, file2_contents = file1.readlines(), file2.readlines()
        assert file1_contents == file2_contents

@pytest.mark.parametrize("object_type", ["body", "interaction", "system"])
@pytest.mark.parametrize("place", ["root", "path"])
@pytest.mark.parametrize("indent_type", ["indent", "noindent"])
def test_to_json(object_type, place, indent_type):
    """Ensure object export to JSON produces files identical to the control."""
    if object_type == "body":
        obj = make_body_for_json()
    elif object_type == "interaction":
        obj = make_interaction_for_json()
    elif object_type == "system":
        obj = make_system_for_json()

    control_path = Path(REFERENCE_FOLDER) / f"{object_type}_at_{place}_{indent_type}.json"
    indent = None if indent_type == "noindent" else 2

    with tempfile.TemporaryDirectory(dir=REFERENCE_FOLDER) as tempdir:
        test_path = Path(tempdir) / f"test_{object_type}_at_{place}_{indent_type}.json"
        
        if place == "root":
            obj.to_json(test_path, json_path=".", indent=indent)
        
        else:
            # In order for json_path to work, the file must already exist, so
            # make one with the same name
            with open(test_path, "w") as f:
                json.dump({"something else": "is here"}, f, indent=indent)
            obj.to_json(test_path, json_path=f"path.to.{object_type}", indent=indent)
        
        compare_text_files(control_path, test_path)

@pytest.mark.parametrize("object_type", ["body", "interaction", "system"])
@pytest.mark.parametrize("place", ["root", "path"])
@pytest.mark.parametrize("indent_type", ["indent", "noindent"])
def test_from_json(object_type, place, indent_type):
    """Ensure object creation from JSON produces the expected result."""
    if object_type == "body":
        obj = make_body_for_json()
    elif object_type == "interaction":
        obj = make_interaction_for_json()
    elif object_type == "system":
        obj = make_system_for_json()

    control_path = Path(REFERENCE_FOLDER) / f"{object_type}_at_{place}_{indent_type}.json"
    json_path = "." if place == "root" else f"path.to.{object_type}"

    if object_type == "body":
        test_obj = p4.Body.from_json(control_path, json_path)
    elif object_type == "interaction":
        test_obj = p4.Interaction.from_json(control_path, json_path)
    elif object_type == "system":
        test_obj = p4.System.from_json(control_path, json_path)

    assert obj == test_obj
