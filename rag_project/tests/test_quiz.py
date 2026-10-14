import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("pydantic"), "Pydantic is not installed")
class QuizTests(unittest.TestCase):
    def test_question_requires_four_unique_options_and_answer_in_range(self):
        from core.quiz_service import QuizQuestion
        from pydantic import ValidationError

        base = {
            "question": "Which process releases energy?",
            "options": ["Respiration", "Diffusion", "Osmosis", "Transpiration"],
            "answer_index": 0,
            "explanation": "The notes state that respiration releases energy.",
            "evidence": "Respiration releases energy",
            "source_ids": ["S1"],
        }
        self.assertEqual(QuizQuestion.model_validate(base).answer_index, 0)
        with self.assertRaises(ValidationError):
            QuizQuestion.model_validate({**base, "options": ["same", "same", "third", "fourth"]})
        with self.assertRaises(ValidationError):
            QuizQuestion.model_validate({**base, "answer_index": 4})
        with self.assertRaises(ValidationError):
            QuizQuestion.model_validate({**base, "answer_index": "0"})

    def test_scoring_counts_unanswered_as_incorrect(self):
        from core.quiz_service import QuizQuestion, score_quiz

        question = QuizQuestion(
            question="Which process releases energy?",
            options=["Respiration", "Diffusion", "Osmosis", "Transpiration"],
            answer_index=0,
            explanation="The notes state that respiration releases energy.",
            evidence="Respiration releases energy",
            source_ids=["S1"],
        )
        self.assertEqual(score_quiz((question,), {0: None}), (0, 1))
        self.assertEqual(score_quiz((question,), {0: 0}), (1, 1))

    def test_malformed_model_json_is_rejected(self):
        from core.errors import GenerationError
        from core.quiz_service import generate_quiz
        from core.retrieval import Passage

        class FakeStore:
            def search(self, query, **kwargs):
                return [Passage("S1", "doc", "notes.txt", None, "Cell respiration releases energy.", 0.9)]

        class FakeLLM:
            def generate(self, *args, **kwargs):
                return "This is not JSON."

        with self.assertRaises(GenerationError):
            generate_quiz(
                "cell respiration",
                FakeStore(),
                FakeLLM(),
                "model",
                document_ids=["doc"],
                question_count=1,
            )

    def test_quiz_answer_must_be_supported_by_quoted_evidence(self):
        from core.errors import GenerationError
        from core.quiz_service import generate_quiz
        from core.retrieval import Passage

        class FakeStore:
            def search(self, query, **kwargs):
                return [Passage("S1", "doc", "notes.txt", None, "Cell respiration releases energy.", 0.9)]

        class FakeLLM:
            def generate(self, *args, **kwargs):
                return (
                    '{"questions":[{"question":"Which process releases energy?",'
                    '"options":["Photosynthesis","Diffusion","Osmosis","Transpiration"],'
                    '"answer_index":0,"explanation":"This process releases energy.",'
                    '"evidence":"Cell respiration releases energy.","source_ids":["S1"]}]}'
                )

        with self.assertRaises(GenerationError):
            generate_quiz(
                "cell respiration",
                FakeStore(),
                FakeLLM(),
                "model",
                document_ids=["doc"],
                question_count=1,
            )

    def test_retries_once_when_correct_answer_is_not_copied_from_evidence(self):
        from core.quiz_service import generate_quiz
        from core.retrieval import Passage

        class FakeStore:
            def search(self, query, **kwargs):
                return [
                    Passage(
                        "S1",
                        "doc",
                        "notes.txt",
                        None,
                        "RAG combines retrieval with language generation.",
                        0.9,
                    )
                ]

        invalid = (
            '{"questions":[{"question":"What does RAG combine?",'
            '"options":["Search and generation","Plants","Osmosis","Diffusion"],'
            '"answer_index":0,"explanation":"The passage describes both parts.",'
            '"evidence":"RAG combines retrieval with language generation.",'
            '"source_ids":["S1"]}]}'
        )
        valid = (
            '{"questions":[{"question":"What does RAG combine?",'
            '"options":["retrieval","Plants","Osmosis","Diffusion"],'
            '"answer_index":0,"explanation":"The passage names retrieval.",'
            '"evidence":"RAG combines retrieval with language generation.",'
            '"source_ids":["S1"]}]}'
        )

        class FakeLLM:
            def __init__(self):
                self.responses = iter([invalid, valid])
                self.prompts = []

            def generate(self, model, prompt, **kwargs):
                self.prompts.append(prompt)
                return next(self.responses)

        llm = FakeLLM()
        questions = generate_quiz(
            "RAG",
            FakeStore(),
            llm,
            "model",
            document_ids=["doc"],
            question_count=1,
        )
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].options[questions[0].answer_index], "retrieval")
        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("correct answer is not present", llm.prompts[1])

    def test_replaces_repeated_distractors_with_other_generated_options(self):
        from core.quiz_service import generate_quiz
        from core.retrieval import Passage

        class FakeStore:
            def search(self, query, **kwargs):
                return [
                    Passage(
                        "S1",
                        "doc",
                        "notes.txt",
                        None,
                        "RAG combines retrieval with language generation.",
                        0.9,
                    )
                ]

        class FakeLLM:
            def __init__(self):
                self.responses = iter(
                    [
                        (
                            '{"questions":[{"question":"What does RAG combine?",'
                            '"options":["retrieval","Plants","Osmosis","Diffusion"],'
                            '"answer_index":0,"explanation":"The passage names retrieval.",'
                            '"evidence":"RAG combines retrieval with language generation.",'
                            '"source_ids":["S1"]}]}'
                        ),
                        (
                            '{"questions":[{"question":"What else does RAG combine?",'
                            '"options":["generation","Plants","Plants","Diffusion"],'
                            '"answer_index":0,"explanation":"The passage names generation.",'
                            '"evidence":"RAG combines retrieval with language generation.",'
                            '"source_ids":["S1"]}]}'
                        ),
                    ]
                )

            def generate(self, *args, **kwargs):
                return next(self.responses)

        questions = generate_quiz(
            "RAG",
            FakeStore(),
            FakeLLM(),
            "model",
            document_ids=["doc"],
            question_count=2,
        )
        self.assertEqual(len(set(questions[0].options)), 4)
        self.assertEqual(questions[0].options[questions[0].answer_index], "retrieval")
        self.assertEqual(len(set(questions[1].options)), 4)
        self.assertEqual(questions[1].options[questions[1].answer_index], "generation")

    def test_retries_with_a_different_fact_when_model_repeats_a_question(self):
        from core.quiz_service import generate_quiz
        from core.retrieval import Passage

        class FakeStore:
            def search(self, query, **kwargs):
                return [
                    Passage(
                        "S1",
                        "doc",
                        "notes.txt",
                        None,
                        "RAG combines retrieval with language generation. "
                        "The retriever searches a knowledge base for relevant passages.",
                        0.9,
                    ),
                    Passage(
                        "S2",
                        "doc",
                        "notes.txt",
                        None,
                        "RAG combines retrieval with language generation.",
                        0.8,
                    ),
                    Passage(
                        "S3",
                        "doc",
                        "notes.txt",
                        None,
                        "The retriever searches a knowledge base for relevant passages.",
                        0.7,
                    ),
                ]

        first_question = (
            '{"questions":[{"question":"What does RAG combine?",'
            '"options":["retrieval","Plants","Osmosis","Diffusion"],'
            '"answer_index":0,"explanation":"The passage names retrieval.",'
            '"evidence":"RAG combines retrieval with language generation.",'
            '"source_ids":["S1"]}]}'
        )
        repeated_question = first_question.replace('"S1"', '"S2"')
        different = (
            '{"questions":[{"question":"What does the retriever search?",'
            '"options":["knowledge base","Plants","Osmosis","Diffusion"],'
            '"answer_index":0,"explanation":"The passage names a knowledge base.",'
            '"evidence":"The retriever searches a knowledge base for relevant passages.",'
            '"source_ids":["S3"]}]}'
        )

        class FakeLLM:
            def __init__(self):
                self.responses = iter(
                    [first_question, repeated_question, different]
                )
                self.prompts = []

            def generate(self, model, prompt, **kwargs):
                self.prompts.append(prompt)
                return next(self.responses)

        llm = FakeLLM()
        questions = generate_quiz(
            "RAG",
            FakeStore(),
            llm,
            "model",
            document_ids=["doc"],
            question_count=2,
        )

        self.assertEqual(len(questions), 2)
        self.assertEqual(questions[0].question, "What does RAG combine?")
        self.assertNotEqual(questions[0].question, questions[1].question)
        self.assertEqual(
            questions[1].options[questions[1].answer_index], "knowledge base"
        )
        self.assertEqual(questions[1].source_ids, ["S3"])
        self.assertEqual(len(llm.prompts), 3)
        self.assertIn("repeated an earlier question", llm.prompts[2])
        self.assertIn("The retriever searches", llm.prompts[2])
        self.assertNotIn("RAG combines retrieval", llm.prompts[2])

    def test_uses_unseen_source_passages_for_later_questions(self):
        from core.quiz_service import generate_quiz
        from core.retrieval import Passage

        first_passage = Passage(
            "S1",
            "doc",
            "notes.txt",
            None,
            "RAG combines retrieval with language generation.",
            0.9,
        )
        second_passage = Passage(
            "S2",
            "doc",
            "notes.txt",
            None,
            "A retriever searches a knowledge base for relevant passages.",
            0.8,
        )

        class FakeStore:
            def __init__(self):
                self.requested_count = None

            def search(self, query, **kwargs):
                self.requested_count = kwargs["count"]
                return [first_passage, second_passage]

        first_question = (
            '{"questions":[{"question":"What does RAG combine?",'
            '"options":["retrieval","Plants","Osmosis","Diffusion"],'
            '"answer_index":0,"explanation":"The passage names retrieval.",'
            '"evidence":"RAG combines retrieval with language generation.",'
            '"source_ids":["S1"]}]}'
        )
        second_question = (
            '{"questions":[{"question":"What does a retriever search?",'
            '"options":["knowledge base","Plants","Osmosis","Diffusion"],'
            '"answer_index":0,"explanation":"The passage names a knowledge base.",'
            '"evidence":"A retriever searches a knowledge base for relevant passages.",'
            '"source_ids":["S2"]}]}'
        )

        class FakeLLM:
            def __init__(self):
                self.responses = iter(
                    [first_question, first_question, second_question]
                )
                self.prompts = []

            def generate(self, *args, **kwargs):
                self.prompts.append(args[1])
                return next(self.responses)

        llm = FakeLLM()
        store = FakeStore()
        questions = generate_quiz(
            "RAG",
            store,
            llm,
            "model",
            document_ids=["doc"],
            question_count=2,
            retrieval_count=1,
        )

        self.assertEqual(questions[0].source_ids, ["S1"])
        self.assertEqual(questions[1].source_ids, ["S2"])
        self.assertNotEqual(questions[0].question, questions[1].question)
        self.assertEqual(store.requested_count, 4)
        self.assertNotIn(first_passage.text, llm.prompts[1])
        self.assertIn(second_passage.text, llm.prompts[1])

    def test_retries_beyond_five_passages_for_later_questions(self):
        from core.quiz_service import generate_quiz
        from core.retrieval import Passage

        passages = [
            Passage(
                f"S{index}",
                "doc",
                "notes.txt",
                None,
                f"Fact {index} is a supported item in chatbot notes.",
                0.9 - index / 100,
            )
            for index in range(1, 10)
        ]

        class FakeStore:
            def search(self, query, **kwargs):
                return passages

        def response(index):
            return (
                '{"questions":[{"question":"What is listed as item '
                f'{index}?","options":["Fact {index}","Plants","Osmosis",'
                '"Diffusion"],"answer_index":0,'
                f'"explanation":"The notes identify Fact {index}.",'
                f'"evidence":"{passages[index - 1].text}",'
                f'"source_ids":["S{index}"]}}]}}'
            )

        class FakeLLM:
            def __init__(self):
                self.responses = iter(
                    [response(1), response(2), response(3)]
                    + ["invalid"] * 5
                    + [response(9)]
                )
                self.prompts = []

            def generate(self, model, prompt, **kwargs):
                self.prompts.append(prompt)
                return next(self.responses)

        llm = FakeLLM()
        questions = generate_quiz(
            "chatbot",
            FakeStore(),
            llm,
            "model",
            document_ids=["doc"],
            question_count=4,
            retrieval_count=5,
        )

        self.assertEqual(len(questions), 4)
        self.assertEqual(questions[-1].source_ids, ["S9"])
        self.assertEqual(len(llm.prompts), 9)
        self.assertIn(passages[8].text, llm.prompts[-1])

    def test_evidence_is_quoted_from_the_selected_source(self):
        from core.quiz_service import generate_quiz
        from core.retrieval import Passage

        passage = Passage(
            "S1",
            "doc",
            "notes.txt",
            None,
            "RAG combines retrieval with language generation.",
            0.9,
        )

        class FakeStore:
            def search(self, query, **kwargs):
                return [passage]

        class FakeLLM:
            def generate(self, *args, **kwargs):
                return (
                    '{"questions":[{"question":"What does RAG combine?",'
                    '"options":["retrieval","Plants","Osmosis","Diffusion"],'
                    '"answer_index":0,"explanation":"The passage names retrieval.",'
                    '"evidence":"RAG uses searching alongside generated text.",'
                    '"source_ids":["S1"]}]}'
                )

        questions = generate_quiz(
            "RAG",
            FakeStore(),
            FakeLLM(),
            "model",
            document_ids=["doc"],
            question_count=1,
        )
        self.assertEqual(questions[0].evidence, passage.text)

    def test_answer_is_cited_to_the_retrieved_source_that_supports_it(self):
        from core.quiz_service import generate_quiz
        from core.retrieval import Passage

        unsupported = Passage("S1", "doc", "notes.txt", None, "A general overview.", 0.9)
        supporting = Passage(
            "S2",
            "doc",
            "notes.txt",
            None,
            "RAG means Retrieval-Augmented Generation.",
            0.8,
        )

        class FakeStore:
            def search(self, query, **kwargs):
                return [unsupported, supporting]

        class FakeLLM:
            def generate(self, *args, **kwargs):
                return (
                    '{"questions":[{"question":"What does RAG mean?",'
                    '"options":["Retrieval-Augmented Generation","Plants","Osmosis",'
                    '"Diffusion"],"answer_index":0,'
                    '"explanation":"The passage defines RAG.",'
                    '"evidence":"Not quoted from the selected source.",'
                    '"source_ids":["S1"]}]}'
                )

        questions = generate_quiz(
            "RAG",
            FakeStore(),
            FakeLLM(),
            "model",
            document_ids=["doc"],
            question_count=1,
        )
        self.assertEqual(questions[0].source_ids, ["S2"])
        self.assertEqual(questions[0].evidence, supporting.text)


if __name__ == "__main__":
    unittest.main()
