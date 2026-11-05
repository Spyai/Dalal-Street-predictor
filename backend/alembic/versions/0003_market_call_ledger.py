"""market call ledger, grades and model registry (Phase 2)

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-31 10:45:00
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


# The ledger is append-only (brief §5): a trigger rejects UPDATE and DELETE on
# `prediction`. TRUNCATE is deliberately not blocked, so test databases can reset.
IMMUTABLE_FN = """
CREATE OR REPLACE FUNCTION prediction_immutable() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'prediction ledger is append-only: % on prediction id % rejected',
        TG_OP, OLD.id USING ERRCODE = 'restrict_violation';
END;
$$ LANGUAGE plpgsql;
"""


def upgrade() -> None:
    op.create_table(
        "model_version",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=32), nullable=False),
        sa.Column("target", sa.String(length=32), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("feature_names", sa.JSON(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("artifact", sa.Text(), nullable=True),
        sa.Column("trained_through", sa.Date(), nullable=False),
        sa.Column("train_rows", sa.Integer(), nullable=False),
        sa.Column("is_champion", sa.Boolean(), nullable=False),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("promotion_note", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "name", "target", "version", name="uq_model_version_name_target_version"
        ),
    )
    op.create_index(op.f("ix_model_version_name"), "model_version", ["name"], unique=False)
    # At most one champion per (name, target).
    op.create_index(
        "uq_model_version_champion",
        "model_version",
        ["name", "target"],
        unique=True,
        postgresql_where=sa.text("is_champion"),
    )

    op.create_table(
        "prediction",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("target", sa.String(length=32), nullable=False),
        sa.Column("session_date", sa.Date(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_late", sa.Boolean(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=True),
        sa.Column("reference_date", sa.Date(), nullable=False),
        sa.Column("reference_close", sa.Float(), nullable=False),
        sa.Column("prob_up", sa.Float(), nullable=False),
        sa.Column("prob_flat", sa.Float(), nullable=False),
        sa.Column("prob_down", sa.Float(), nullable=False),
        sa.Column("predicted_class", sa.String(length=8), nullable=False),
        sa.Column("range_p10_pct", sa.Float(), nullable=True),
        sa.Column("range_p90_pct", sa.Float(), nullable=True),
        sa.Column("gap_call", sa.String(length=8), nullable=True),
        sa.Column("gap_expected_pct", sa.Float(), nullable=True),
        sa.Column("confidence", sa.String(length=8), nullable=False),
        sa.Column("no_edge", sa.Boolean(), nullable=False),
        sa.Column("drivers", sa.JSON(), nullable=False),
        sa.Column("timing_guidance", sa.Text(), nullable=True),
        sa.Column("baselines", sa.JSON(), nullable=False),
        sa.Column("feature_snapshot", sa.JSON(), nullable=False),
        sa.Column("corrects_prediction_id", sa.BigInteger(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("prev_hash", sa.String(length=64), nullable=False),
        sa.Column("hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["corrects_prediction_id"], ["prediction.id"]),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_version.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("hash"),
        sa.UniqueConstraint("seq"),
    )
    op.create_index(
        op.f("ix_prediction_session_date"), "prediction", ["session_date"], unique=False
    )
    op.create_index(op.f("ix_prediction_target"), "prediction", ["target"], unique=False)
    op.execute(IMMUTABLE_FN)
    op.execute(
        "CREATE TRIGGER prediction_immutable BEFORE UPDATE OR DELETE ON prediction "
        "FOR EACH ROW EXECUTE FUNCTION prediction_immutable()"
    )

    op.create_table(
        "prediction_grade",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("prediction_id", sa.BigInteger(), nullable=False),
        sa.Column("graded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actual_open", sa.Float(), nullable=True),
        sa.Column("actual_close", sa.Float(), nullable=True),
        sa.Column("actual_pct", sa.Float(), nullable=True),
        sa.Column("actual_class", sa.String(length=8), nullable=True),
        sa.Column("correct", sa.Boolean(), nullable=True),
        sa.Column("brier", sa.Float(), nullable=True),
        sa.Column("log_loss", sa.Float(), nullable=True),
        sa.Column("gap_actual_pct", sa.Float(), nullable=True),
        sa.Column("gap_correct", sa.Boolean(), nullable=True),
        sa.Column("gap_graded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("baseline_correct", sa.JSON(), nullable=True),
        sa.Column("error_category", sa.String(length=32), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(["prediction_id"], ["prediction.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("prediction_id", name="uq_prediction_grade_prediction"),
    )
    op.create_index(
        op.f("ix_prediction_grade_prediction_id"),
        "prediction_grade",
        ["prediction_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_prediction_grade_prediction_id"), table_name="prediction_grade")
    op.drop_table("prediction_grade")
    op.execute("DROP TRIGGER IF EXISTS prediction_immutable ON prediction")
    op.execute("DROP FUNCTION IF EXISTS prediction_immutable()")
    op.drop_index(op.f("ix_prediction_target"), table_name="prediction")
    op.drop_index(op.f("ix_prediction_session_date"), table_name="prediction")
    op.drop_table("prediction")
    op.drop_index("uq_model_version_champion", table_name="model_version")
    op.drop_index(op.f("ix_model_version_name"), table_name="model_version")
    op.drop_table("model_version")
