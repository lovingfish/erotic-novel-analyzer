from backend import merge_analysis_results


def test_merge_analysis_results_merges_by_keys_and_fills_fields():
    original = {
        "characters": [
            {"name": "女主", "gender": "female", "identity": "", "sexual_preferences": ""},
        ],
        "relationships": [
            {"from": "女主", "to": "男主", "type": "恋人", "description": ""},
        ],
        "first_sex_scenes": [
            {"participants": ["女主", "男主"], "chapter": "第1章", "location": "", "description": ""},
        ],
        "sex_scenes": {
            "total_count": 1,
            "scenes": [
                {"chapter": "第1章", "participants": ["女主", "男主"], "location": "", "description": ""},
            ],
        },
        "evolution": [
            {"chapter": "第1章", "stage": "开始", "description": ""},
        ],
        "thunderzones": [
            {"type": "NTR", "chapter_location": "第2章", "severity": "", "description": ""},
        ],
    }
    verification = {
        "missing_characters": [
            {"name": "女主", "identity": "学生", "sexual_preferences": "偏爱口交"},
            {"name": "男主", "gender": "male", "identity": "丈夫"},
        ],
        "missing_relationships": [
            {"from": "女主", "to": "男主", "description": "更完整的关系描述"},
        ],
        "missing_first_sex_scenes": [
            {"participants": ["男主", "女主"], "location": "卧室", "description": "第一次"},
        ],
        "missing_sex_scenes": [
            {"chapter": "第2章", "participants": ["男主", "女主"], "location": "浴室", "description": "新增场景"},
        ],
        "missing_evolution": [
            {"chapter": "第1章", "description": "补充发展"},
        ],
        "missing_thunderzones": [
            {"type": "NTR", "chapter_location": "第2章", "severity": "高", "description": "补充雷点"},
        ],
    }

    merged = merge_analysis_results(original, verification)

    names = {c.get("name") for c in merged.get("characters") or []}
    assert names == {"女主", "男主"}

    heroine = next(c for c in merged["characters"] if c.get("name") == "女主")
    assert heroine.get("identity") == "学生"
    assert heroine.get("sexual_preferences") == "偏爱口交"

    rel = next(r for r in merged["relationships"] if r.get("from") == "女主" and r.get("to") == "男主")
    assert rel.get("description") == "更完整的关系描述"

    assert len(merged.get("first_sex_scenes") or []) == 1
    first = merged["first_sex_scenes"][0]
    assert first.get("location") == "卧室"
    assert first.get("description") == "第一次"

    assert len((merged.get("sex_scenes") or {}).get("scenes") or []) == 2
    assert merged["sex_scenes"]["total_count"] >= 2

    evo = next(e for e in merged.get("evolution") or [] if e.get("chapter") == "第1章")
    assert evo.get("description") == "补充发展"

    th = next(t for t in merged.get("thunderzones") or [] if t.get("type") == "NTR" and t.get("chapter_location") == "第2章")
    assert th.get("severity") == "高"
    assert th.get("description") == "补充雷点"


def test_merge_analysis_results_dedups_scenes_by_chapter_and_participants():
    original = {
        "first_sex_scenes": [
            {"participants": ["A", "B"], "chapter": "1", "location": "", "description": ""},
        ],
        "sex_scenes": {
            "total_count": 1,
            "scenes": [
                {"chapter": "1", "participants": ["A", "B"], "location": "", "description": ""},
            ],
        },
    }
    verification = {
        "missing_first_sex_scenes": [
            {"participants": ["B", "A"], "location": "L1", "description": "D1"},
        ],
        "missing_sex_scenes": [
            {"chapter": "1", "participants": ["B", "A"], "location": "L1", "description": "D1"},
            {"chapter": "2", "participants": ["A", "B"], "location": "L2", "description": "D2"},
        ],
    }

    merged = merge_analysis_results(original, verification)

    assert len(merged["first_sex_scenes"]) == 1
    assert merged["first_sex_scenes"][0]["location"] == "L1"
    assert merged["first_sex_scenes"][0]["description"] == "D1"

    scenes = merged["sex_scenes"]["scenes"]
    assert len(scenes) == 2
    scene1 = next(s for s in scenes if s.get("chapter") == "1")
    assert scene1.get("location") == "L1"
    assert scene1.get("description") == "D1"
