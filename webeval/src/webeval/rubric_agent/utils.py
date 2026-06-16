"""Shared rubric-validation helpers.

Carved out of ``mm_rubric_agent.py`` so both ``mm_rubric_agent`` and its
``HumanFeedbackAgent`` mixin can import them without round-tripping
through the partially-initialised ``mm_rubric_agent`` module (which
previously caused circular-import headaches).

These functions are pure and stateless — no agent / config dependency.
"""

from __future__ import annotations


def verify_rubric(d: dict) -> bool:
    assert isinstance(d, dict), f"Expected a dict, got {type(d)}"
    assert "items" in d, "Expected 'items' field in dict"
    assert isinstance(d["items"], list), "Expected 'items' field to be a list"
    for item in d["items"]:
        assert "criterion" in item, "Expected 'criterion' field in each item"
        if "items" in item:
            verify_rubric(item)
        else:
            assert "max_points" in item, "Expected 'max_points' field in each item"
            assert isinstance(
                item["max_points"], (int, float)
            ), "'max_points' should be a number"
            assert (
                "earned_points" in item
            ), "Expected 'earned_points' field in each item"
            assert isinstance(
                item["earned_points"], (int, float)
            ), "'earned_points' should be a number"
            assert (
                "justification" in item
            ), "Expected 'justification' field in each item"
            assert (
                isinstance(item["justification"], str) and item["justification"]
            ), "'justification' should be a string"
            if "condition" in item:
                assert (
                    "is_condition_met" in item
                ), f"Conditional criterion '{item['criterion']}' must have 'is_condition_met' field"
                assert isinstance(
                    item["is_condition_met"], bool
                ), f"'is_condition_met' must be a boolean for criterion '{item['criterion']}'"
            if "post_image_justification" in item:
                assert (
                    isinstance(item["post_image_justification"], str)
                    and item["post_image_justification"]
                ), "'post_image_justification' should be a non-empty string"
            if "post_image_earned_points" in item:
                assert isinstance(
                    item["post_image_earned_points"], (int, float)
                ), "'post_image_earned_points' should be a number"
                assert (
                    0 <= item["post_image_earned_points"] <= item["max_points"]
                ), f"'post_image_earned_points' ({item['post_image_earned_points']}) must be between 0 and max_points ({item['max_points']})"
    return True


def verify_generated_rubric(d: dict) -> bool:
    assert isinstance(d, dict), f"Expected a dict, got {type(d)}"
    assert "items" in d, "Expected 'items' field in dict"
    assert isinstance(d["items"], list), "Expected 'items' field to be a list"
    assert len(d["items"]) > 0, "Expected at least one item in rubric"
    for item in d["items"]:
        assert "criterion" in item, "Expected 'criterion' field in each item"
        assert "description" in item, "Expected 'description' field in each item"
        assert "max_points" in item, "Expected 'max_points' field in each item"
        assert isinstance(
            item["max_points"], (int, float)
        ), "'max_points' should be a number"
        assert item["max_points"] > 0, "'max_points' should be greater than 0"
        assert "justification" in item, "Expected 'justification' field in each item"
        assert "earned_points" in item, "Expected 'earned_points' field in each item"
        assert (
            item["justification"] == ""
        ), "'justification' should be empty string in generated rubric"
        assert (
            item["earned_points"] == ""
        ), "'earned_points' should be empty string in generated rubric"
        if "items" in item:
            verify_generated_rubric(item)
    return True


def verify_conditional_totals(d: dict) -> bool:
    """Verify that total_max_points and total_earned_points correctly account for conditional criteria.

    Rules:
    - Non-conditional criteria: Always count max_points and earned_points toward totals
    - Conditional criteria with is_condition_met=true: Count max_points and earned_points toward totals
    - Conditional criteria with is_condition_met=false: Do NOT count toward totals (excluded from both numerator and denominator)
    """
    assert isinstance(d, dict), f"Expected a dict, got {type(d)}"
    assert "items" in d, "Expected 'items' field in dict"
    assert "total_max_points" in d, "Expected 'total_max_points' field in dict"
    assert "total_earned_points" in d, "Expected 'total_earned_points' field in dict"

    def sum_points_recursive(items, breakdown_list):
        total_max = 0
        total_earned = 0

        for item in items:
            if "items" in item:
                sub_max, sub_earned = sum_points_recursive(
                    item["items"], breakdown_list
                )
                total_max += sub_max
                total_earned += sub_earned
            else:
                is_conditional = "condition" in item
                criterion_name = item.get("criterion", "unnamed")

                if is_conditional:
                    assert (
                        "is_condition_met" in item
                    ), f"Conditional criterion '{criterion_name}' missing 'is_condition_met' field"

                    if item["is_condition_met"]:
                        total_max += item["max_points"]
                        total_earned += item["earned_points"]
                        breakdown_list.append(
                            f"  COUNTED (conditional, condition met): '{criterion_name}' "
                            f"[max: {item['max_points']}, earned: {item['earned_points']}]"
                        )
                    else:
                        breakdown_list.append(
                            f"  EXCLUDED (conditional, condition NOT met): '{criterion_name}' "
                            f"[max: {item['max_points']}, earned: {item['earned_points']}] - NOT counted in totals"
                        )
                else:
                    total_max += item["max_points"]
                    total_earned += item["earned_points"]
                    breakdown_list.append(
                        f"  COUNTED (non-conditional): '{criterion_name}' "
                        f"[max: {item['max_points']}, earned: {item['earned_points']}]"
                    )

        return total_max, total_earned

    breakdown = []
    expected_max, expected_earned = sum_points_recursive(d["items"], breakdown)

    max_matches = abs(d["total_max_points"] - expected_max) < 0.01
    earned_matches = abs(d["total_earned_points"] - expected_earned) < 0.01

    if not max_matches or not earned_matches:
        error_msg = [
            "\n" + "=" * 80,
            "ERROR: Total points calculation does not follow conditional criteria rules!",
            "=" * 80,
            "",
            "RULES REMINDER:",
            "  1. Non-conditional criteria: ALWAYS count max_points and earned_points",
            "  2. Conditional criteria (has 'condition' field):",
            "     - If is_condition_met = true: COUNT the points",
            "     - If is_condition_met = false: DO NOT COUNT (exclude from both numerator and denominator)",
            "",
            "BREAKDOWN OF ALL CRITERIA:",
        ]
        error_msg.extend(breakdown)
        error_msg.extend(
            [
                "",
                "CALCULATION SUMMARY:",
                f"  Expected total_max_points:    {expected_max}",
                f"  Reported total_max_points:    {d['total_max_points']}",
                f"  Match: {'YES' if max_matches else 'NO - MISMATCH!'}",
                "",
                f"  Expected total_earned_points: {expected_earned}",
                f"  Reported total_earned_points: {d['total_earned_points']}",
                f"  Match: {'YES' if earned_matches else 'NO - MISMATCH!'}",
                "",
                "REQUIRED FIX:",
            ]
        )

        if not max_matches:
            error_msg.append(
                f"  - Change 'total_max_points' from {d['total_max_points']} to {expected_max}"
            )
        if not earned_matches:
            error_msg.append(
                f"  - Change 'total_earned_points' from {d['total_earned_points']} to {expected_earned}"
            )

        error_msg.extend(
            [
                "",
                "=" * 80,
            ]
        )

        raise AssertionError("\n".join(error_msg))

    return True
