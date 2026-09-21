"""协议层单元测试（零依赖，不需模型/GPU/网络）。"""
import unittest

from jevp.protocol import (
    Answer,
    ChoiceQuestion,
    NoulQuestion,
    Question,
    build_prompt,
    parse_question,
    validate_questions,
)


class TestQuestionValidation(unittest.TestCase):
    def test_noul_minimal(self):
        q = Question(type="noul", instructions="紧急吗？")
        q.validate()  # 不抛异常

    def test_noul_bad_criteria(self):
        q = Question(type="noul", instructions="x", criteria={"yes": "y"})
        with self.assertRaises(ValueError):
            q.validate()

    def test_choice_requires_two(self):
        q = Question(type="choice", instructions="x", criteria={"a": "A"})
        with self.assertRaises(ValueError):
            q.validate()
        q = Question(type="choice", instructions="x", criteria={"a": "A", "b": "B"})
        q.validate()

    def test_score_requires_two(self):
        q = Question(type="score", instructions="x", criteria=["only"])
        with self.assertRaises(ValueError):
            q.validate()

    def test_unknown_type(self):
        q = Question(type="weird", instructions="x")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            q.validate()


class TestParseAndRoundtrip(unittest.TestCase):
    def test_parse_roundtrip(self):
        raw = {
            "type": "choice",
            "instructions": "route to?",
            "criteria": {"billing": "pay", "tech": "bugs"},
        }
        q = parse_question(raw)
        self.assertEqual(q.to_dict(), raw)

    def test_candidate_labels(self):
        q = ChoiceQuestion(instructions="x", criteria={"a": "A", "b": "B", "c": "C"})
        self.assertEqual(q.candidate_labels(), ["a", "b", "c"])
        s = Question(type="score", instructions="x", criteria=["低", "中", "高"])
        self.assertEqual(s.candidate_labels(), ["0", "1", "2"])

    def test_validate_questions_group(self):
        qs = validate_questions({
            "a": {"type": "noul", "instructions": "x"},
            "b": {"type": "choice", "instructions": "x", "criteria": {"p": "P", "q": "Q"}},
        })
        self.assertEqual(set(qs), {"a", "b"})
        with self.assertRaises(ValueError):
            validate_questions({})


class TestPrompt(unittest.TestCase):
    def test_prompt_contains_state_and_instruction(self):
        q = ChoiceQuestion(instructions="哪个部门？", criteria={"billing": "计费", "tech": "技术"})
        p = build_prompt({"ticket": "打不开"}, "dept", q)
        self.assertIn("打不开", p)
        self.assertIn("哪个部门？", p)
        self.assertIn("billing: 计费", p)

    def test_prompt_state_json_stringifed(self):
        q = NoulQuestion(instructions="紧急吗？")
        p = build_prompt({"a": 1}, "u", q)
        self.assertIn('"a": 1', p)


class TestAnswer(unittest.TestCase):
    def test_to_dict(self):
        a = Answer("noul", {"noul": 0.9, "raw_mass": 0.3})
        self.assertEqual(a.to_dict(), {"type": "noul", "noul": 0.9, "raw_mass": 0.3})


if __name__ == "__main__":
    unittest.main()