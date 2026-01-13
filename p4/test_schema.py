from p4.body import BodySchema

INVALID_KWARGS = [
    # Secondary types given with no orientations, position keys are missing
    dict(   # 1 secondary type (no positions given)
        primary_type="A",
        secondary_types=["B"]
    ),
    dict(   # 1 secondary type (empty positions given)
        primary_type="A",
        secondary_types=["B"],
        secondary_positions_by_type=dict()
    ),
    dict(   # 1 secondary type (positions given with wrong key)
        primary_type="A",
        secondary_types=["B"],
        secondary_positions_by_type=dict(
            wrong=[[1,1,1]]
        )
    ),
    dict(   # 2 secondary types (first is wrong)
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_positions_by_type=dict(
            C=[[1,0,0], [0,1,0]]
        )
    ),
    dict(   # 2 secondary types (second is wrong)
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_positions_by_type=dict(
            B=[[1,1,1]]
        )
    ),
    dict(   # 2 secondary types (both are wrong)
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_positions_by_type=dict()
    ),

    # Secondary types with orientations, position keys are missing
    dict(   # 1 secondary type
        primary_type="A",
        secondary_types=["B"],
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0,]]
        )
    ),
    dict(   # 1 secondary type (positions given with wrong key)
        primary_type="A",
        secondary_types=["B"],
        secondary_positions_by_type=dict(
            wrong=[[1,1,1]]
        ),
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0,]]
        )
    ),
    dict(   # 2 secondary types (first is wrong)
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_positions_by_type=dict(
            C=[[1,0,0], [0,1,0]]
        ),
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0]],
            C=[[1,0,0,0], [0,1,0,0]]
        )
    ),
    dict(   # 2 secondary types (second is wrong)
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_positions_by_type=dict(
            B=[[1,1,1]]
        ),
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0]],
            C=[[1,0,0,0], [0,1,0,0]]
        )
    ),
    dict(   # 2 secondary types (both are wrong)
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0]],
            C=[[1,0,0,0], [0,1,0,0]]
        )
    ),

    # Secondary types given with mismatched positions and orientations
    dict(   # 1 secondary type
        primary_type="A",
        secondary_types=["B"],
        secondary_positions_by_type=dict(
            B=[[1,1,1]]
        ),
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0,], [1,1,0,0,]]
        )
    ),
    dict(   # 2 secondary types (first does not match)
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_positions_by_type=dict(
            B=[[1,1,1]],
            C=[[1,0,0], [0,1,0]]
        ),
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0], [1,1,0,0,]],
            C=[[1,0,0,0], [0,1,0,0]]
        )
    ),
    dict(   # 2 secondary types (second does not match)
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_positions_by_type=dict(
            B=[[1,1,1]],
            C=[[1,0,0], [0,1,0]]
        ),
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0]],
            C=[[1,0,0,0]]
        )
    ),
    dict(   # 2 secondary types (both do not match)
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_positions_by_type=dict(
            B=[[1,1,1]],
            C=[[1,0,0], [0,1,0]]
        ),
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0], [1,1,0,0,]],
            C=[[1,0,0,0]]
        )
    )
]

_ = BodySchema().load(INVALID_KWARGS[5])
