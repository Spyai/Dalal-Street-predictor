"""stock ideas ledger, outcomes and paper portfolio (Phase 3)

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-31 17:30:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


# Same rule as the prediction ledger (brief §5): rows are never updated or deleted.
IMMUTABLE_FN = """
CREATE OR REPLACE FUNCTION idea_immutable() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'idea ledger is append-only: % on idea id % rejected',
        TG_OP, OLD.id USING ERRCODE = 'restrict_violation';
END;
$$ LANGUAGE plpgsql;
"""


def upgrade() -> None:
    op.create_table(
        "idea",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("session_date", sa.Date(), nullable=False),
        sa.Column("is_late", sa.Boolean(), nullable=False),
        sa.Column("instrument_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("exchange", sa.String(length=16), nullable=False),
        sa.Column("horizon", sa.String(length=16), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("setup", sa.String(length=32), nullable=False),
        sa.Column("direction", sa.String(length=8), nullable=False),
        sa.Column("entry_low", sa.Float(), nullable=False),
        sa.Column("entry_high", sa.Float(), nullable=False),
        sa.Column("stop", sa.Float(), nullable=False),
        sa.Column("target_1", sa.Float(), nullable=False),
        sa.Column("target_2", sa.Float(), nullable=True),
        sa.Column("p_target", sa.Float(), nullable=False),
        sa.Column("expected_return_low_pct", sa.Float(), nullable=False),
        sa.Column("expected_return_high_pct", sa.Float(), nullable=False),
        sa.Column("risk_reward", sa.Float(), nullable=False),
        sa.Column("ev_pct_after_costs", sa.Float(), nullable=False),
        sa.Column("position_qty", sa.Integer(), nullable=False),
        sa.Column("position_value", sa.Float(), nullable=False),
        sa.Column("risk_amount", sa.Float(), nullable=False),
        sa.Column("reasons", sa.JSON(), nullable=False),
        sa.Column("invalidation", sa.Text(), nullable=False),
        sa.Column("drivers", sa.JSON(), nullable=False),
        sa.Column("feature_snapshot", sa.JSON(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=True),
        sa.Column("strategy", sa.String(length=32), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("surveillance_unknown", sa.Boolean(), nullable=False),
        sa.Column("corrects_idea_id", sa.BigInteger(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("prev_hash", sa.String(length=64), nullable=False),
        sa.Column("hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["corrects_idea_id"], ["idea.id"]),
        sa.ForeignKeyConstraint(["instrument_id"], ["instrument.id"]),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_version.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("hash"),
        sa.UniqueConstraint("seq"),
    )
    op.create_index(op.f("ix_idea_instrument_id"), "idea", ["instrument_id"], unique=False)
    op.create_index(op.f("ix_idea_session_date"), "idea", ["session_date"], unique=False)
    op.create_index(op.f("ix_idea_symbol"), "idea", ["symbol"], unique=False)
    op.execute(IMMUTABLE_FN)
    op.execute(
        "CREATE TRIGGER idea_immutable BEFORE UPDATE OR DELETE ON idea "
        "FOR EACH ROW EXECUTE FUNCTION idea_immutable()"
    )

    op.create_table(
        "idea_outcome",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("idea_id", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("entry_date", sa.Date(), nullable=True),
        sa.Column("entry_price", sa.Float(), nullable=True),
        sa.Column("exit_date", sa.Date(), nullable=True),
        sa.Column("exit_price", sa.Float(), nullable=True),
        sa.Column("gross_return_pct", sa.Float(), nullable=True),
        sa.Column("costs_pct", sa.Float(), nullable=True),
        sa.Column("net_return_pct", sa.Float(), nullable=True),
        sa.Column("r_multiple", sa.Float(), nullable=True),
        sa.Column("days_held", sa.Integer(), nullable=True),
        sa.Column("max_favorable_pct", sa.Float(), nullable=True),
        sa.Column("max_adverse_pct", sa.Float(), nullable=True),
        sa.Column("graded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("error_category", sa.String(length=32), nullable=True),
        sa.ForeignKeyConstraint(["idea_id"], ["idea.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idea_id", name="uq_idea_outcome_idea"),
    )
    op.create_index(op.f("ix_idea_outcome_idea_id"), "idea_outcome", ["idea_id"], unique=False)
    op.create_index(op.f("ix_idea_outcome_status"), "idea_outcome", ["status"], unique=False)

    op.create_table(
        "paper_trade",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("idea_id", sa.BigInteger(), nullable=True),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("exchange", sa.String(length=16), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("qty", sa.Integer(), nullable=False),
        sa.Column("entry_date", sa.Date(), nullable=False),
        sa.Column("entry_price", sa.Float(), nullable=False),
        sa.Column("stop", sa.Float(), nullable=True),
        sa.Column("target", sa.Float(), nullable=True),
        sa.Column("exit_date", sa.Date(), nullable=True),
        sa.Column("exit_price", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=8), nullable=False),
        sa.Column("gross_pnl", sa.Float(), nullable=True),
        sa.Column("costs", sa.Float(), nullable=True),
        sa.Column("net_pnl", sa.Float(), nullable=True),
        sa.Column("r_multiple", sa.Float(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["idea_id"], ["idea.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_paper_trade_idea_id"), "paper_trade", ["idea_id"], unique=False)
    op.create_index(op.f("ix_paper_trade_status"), "paper_trade", ["status"], unique=False)
    op.create_index(op.f("ix_paper_trade_symbol"), "paper_trade", ["symbol"], unique=False)

    op.create_table(
        "paper_equity",
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("equity", sa.Float(), nullable=False),
        sa.Column("cash", sa.Float(), nullable=False),
        sa.Column("invested", sa.Float(), nullable=False),
        sa.Column("open_positions", sa.Integer(), nullable=False),
        sa.Column("nifty_close", sa.Float(), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint("trade_date"),
    )


def downgrade() -> None:
    op.drop_table("paper_equity")
    op.drop_index(op.f("ix_paper_trade_symbol"), table_name="paper_trade")
    op.drop_index(op.f("ix_paper_trade_status"), table_name="paper_trade")
    op.drop_index(op.f("ix_paper_trade_idea_id"), table_name="paper_trade")
    op.drop_table("paper_trade")
    op.drop_index(op.f("ix_idea_outcome_status"), table_name="idea_outcome")
    op.drop_index(op.f("ix_idea_outcome_idea_id"), table_name="idea_outcome")
    op.drop_table("idea_outcome")
    op.execute("DROP TRIGGER IF EXISTS idea_immutable ON idea")
    op.execute("DROP FUNCTION IF EXISTS idea_immutable()")
    op.drop_index(op.f("ix_idea_symbol"), table_name="idea")
    op.drop_index(op.f("ix_idea_session_date"), table_name="idea")
    op.drop_index(op.f("ix_idea_instrument_id"), table_name="idea")
    op.drop_table("idea")
