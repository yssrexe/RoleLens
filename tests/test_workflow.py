import unittest
from src.graph.models import Profile, Requirements, Questions, Question
from src.graph.workflow import analyze, build_workflow, score_profile

class FakeLLM:
    def structured(self, schema, instruction, data):
        if schema is Requirements:
            return Requirements(required_skills=['Python', 'Docker'], minimum_years=5)
        if schema is Profile:
            strong = 'strong' in data['resume']
            return Profile(skills=['Python', 'Docker'] if strong else ['Python'], years_experience=6 if strong else 2)
        return Questions(questions=[Question(gap=data['gaps'][0], question='How would you address this in the role?', what_to_listen_for='Practical evidence')])

class WorkflowTests(unittest.TestCase):
    def test_three_agents_rank_and_generate_gap_questions(self):
        graph = build_workflow(FakeLLM(), lambda job, resumes: [.3, .9])
        result = analyze({'job_description':'Python Docker engineer with five years experience', 'resumes':[
            {'label':'Less evidence','text':'weak candidate resume text'},
            {'label':'More evidence','text':'strong candidate resume text'}]}, graph)
        self.assertEqual(result['stages'], ['resume_parser', 'scorer_ranker', 'interview_q_generator'])
        self.assertEqual([c['candidate_id'] for c in result['candidates']], [2,1])
        self.assertEqual(result['candidates'][0]['questions'], [])
        weak = result['candidates'][1]
        self.assertIn(weak['questions'][0]['gap'], weak['gaps'])
        self.assertAlmostEqual(weak['fit_score'], (.45*.3 + .35*.5 + .1*.4)/.9, places=4)

    def test_missing_requirements_do_not_award_free_points(self):
        score = score_profile(Profile(), Requirements(), .2)
        self.assertEqual(score['fit_score'], .2)
        self.assertEqual(score['score_breakdown'], {'semantic':.2})

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
