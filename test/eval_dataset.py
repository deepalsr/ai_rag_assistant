"""
EVAL DATASET
-------------
Job: a fixed set of question/expected-answer pairs, hand-written from
the actual handbook content, so we can measure system quality
objectively instead of eyeballing one or two test questions.
"""

HAND_WRITTEN_CASES = [
    {
        "question": "How many PTO days do full-time employees accrue per year?",
        "expected_answer": "18 days",
        "category": "leave_policy",
    },
    {
        "question": "How many days of unused PTO can be carried over to the next year?",
        "expected_answer": "5 days",
        "category": "leave_policy",
    },
    {
        "question": "How many days per week can employees work remotely?",
        "expected_answer": "3 days per week",
        "category": "remote_work",
    },
    {
        "question": "How much is the annual equipment stipend for home office setup?",
        "expected_answer": "$500 per year",
        "category": "remote_work",
    },
    {
        "question": "How many weeks of paid parental leave does the primary caregiver get?",
        "expected_answer": "16 weeks",
        "category": "parental_leave",
    },
    {
        "question": "Who is the company's Chief Financial Officer?",
        "expected_answer": "I don't have enough information to answer that.",
        "category": "out_of_scope",  # deliberately NOT in the handbook - tests refusal
    },
]