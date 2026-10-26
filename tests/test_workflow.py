import unittest
from src.graph.models import Profile, Requirements, Questions, Question
from src.graph.workflow import analyze, build_workflow, score_profile
from src.graph.nodes.interview_q_generator import interview_q_generator
from pydantic import ValidationError
from rag_fakes import fake_retrieve

class FakeLLM:
    def structured(self, schema, instruction, data):
        if schema is Requirements:
            return Requirements(required_skills=['Python', 'Docker'], minimum_years=5)
        if schema is Profile:
            strong = 'strong' in data['resume']
            return Profile(skills=['Python', 'Docker'] if strong else ['Python'], years_experience=6 if strong else 2)
        return Questions(questions=[Question(gap=data['gaps'][0], question='How would you address this in the role?', what_to_listen_for='Practical evidence')])

class WorkflowTests(unittest.TestCase):
    def test_question_schema_constrains_gaps_and_rejects_invented_gap(self):
        gaps = ['Skill not evidenced: docker', 'Skill not evidenced: python']
        test = self

        class SchemaCheckingLLM:
            def structured(self, schema, instruction, data):
                definition = schema.model_json_schema()['$defs']['GapQuestion']
                test.assertEqual(definition['properties']['gap']['enum'], gaps)
                question = dict(gap='Invented gap', question='Describe your experience?',
                                what_to_listen_for='Concrete examples')
                with test.assertRaises(ValidationError):
                    schema.model_validate({'questions': [question]})
                question['gap'] = gaps[0]
                return schema.model_validate({'questions': [question]})

        result = interview_q_generator({'candidates': [{'gaps': gaps, 'profile': {}}],
                                       'job_description': 'Docker engineer', 'stages': []},
                                      SchemaCheckingLLM())
        self.assertEqual(result['candidates'][0]['questions'][0]['gap'], gaps[0])

    def test_three_agents_rank_and_generate_gap_questions(self):
        graph = build_workflow(FakeLLM(), fake_retrieve)
        result = analyze({'job_description':'Python Docker engineer with five years experience', 'resumes':[
            {'label':'Less evidence','text':'weak candidate resume text'},
            {'label':'More evidence','text':'strong candidate resume text'}]}, graph)
        self.assertEqual(result['stages'], ['retrieve_candidates', 'resume_parser', 'scorer_ranker', 'interview_q_generator'])
        self.assertEqual([c['candidate_id'] for c in result['candidates']], [2,1])
        self.assertEqual(result['candidates'][0]['questions'][0]['gap'], 'Skill assessment: docker')
        weak = result['candidates'][1]
        self.assertIn(weak['questions'][0]['gap'], weak['gaps'])
        self.assertAlmostEqual(weak['fit_score'], (.45*.3 + .35*.5 + .1*.4)/.9, places=4)

    def test_missing_requirements_do_not_award_free_points(self):
        score = score_profile(Profile(), Requirements(), .2)
        self.assertEqual(score['fit_score'], .2)
        self.assertEqual(score['score_breakdown'], {'semantic':.2})

    def test_placeholder_education_does_not_create_gap_or_penalty(self):
        requirements = Requirements(education=['null', None, 'unknown', ' '])
        score = score_profile(Profile(education=None), requirements, .5)
        self.assertEqual(score['gaps'], [])
        self.assertNotIn('education', score['score_breakdown'])
        self.assertEqual(Requirements(education=['bachelor', 'null']).education, ['bachelor'])

    def test_questions_assess_skills_even_when_only_gap_is_education(self):
        from unittest.mock import Mock
        llm = Mock(wraps=FakeLLM())
        result = interview_q_generator({
            'candidates': [{'gaps': ['Education not evidenced: bachelor'],
                            'matched_skills': ['python'],
                            'profile': {'skills': ['Python'], 'education': ['Some School']}}],
            'requirements': {'required_skills': ['Python']},
            'stages': []}, llm)
        data = llm.structured.call_args.args[2]
        self.assertEqual(data['gaps'], ['Skill assessment: python'])
        self.assertNotIn('education', data['profile'])
        self.assertEqual(result['candidates'][0]['questions'][0]['gap'], 'Skill assessment: python')

    def test_aliases_and_distinct_languages(self):
        score=score_profile(Profile(skills=['JS','C++']),Requirements(required_skills=['javascript','C']),1)
        self.assertEqual(score['matched_skills'],['javascript'])
        self.assertEqual(score['gaps'],['Skill not evidenced: c'])

    def test_unknown_experience_is_a_gap(self):
        score=score_profile(Profile(), Requirements(minimum_years=3), .5)
        self.assertEqual(score['score_breakdown']['experience'],0)
        self.assertTrue(score['gaps'])

    def test_invalid_score_and_input(self):
        with self.assertRaises(ValueError):
            score_profile(Profile(),Requirements(),float('nan'))
        with self.assertRaises(ValueError):
            analyze({'job_description':' '*20,'resumes':[{'label':'a','text':' '*20}]})

if __name__ == '__main__':
    unittest.main()
