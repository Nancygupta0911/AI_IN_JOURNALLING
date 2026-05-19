"""
Generate synthetic test dataset for habit extraction pipeline validation
Creates realistic journal entries with various habit mentions
"""

import pandas as pd
from datetime import datetime, timedelta
import random
from pathlib import Path

# Realistic journal entries with diverse habit mentions
SAMPLE_JOURNALS = [
    {
        "id": "j001",
        "date": "2024-01-15",
        "text": """
        Slept late again, around 2:30am. Couldn't stop scrolling instagram and watching reels. 
        Felt really anxious about the upcoming exam. Didn't study at all today, just procrastinated.
        Had three cups of coffee to stay awake. Need to fix my sleep schedule.
        """
    },
    {
        "id": "j002", 
        "date": "2024-01-16",
        "text": """
        Woke up early at 5:30am! Went for a morning run, felt amazing. Had a healthy breakfast 
        with eggs and toast. Studied for 4 hours straight, finally made progress on the assignment.
        Meditated for 15 minutes before bed. Feeling proud and energetic today.
        """
    },
    {
        "id": "j003",
        "date": "2024-01-17", 
        "text": """
        Overslept and missed my 9am lecture. Felt terrible about it. Went to gym in the evening 
        to clear my head. Worked out for an hour. Met friends for dinner, had a good social time.
        Watched Netflix till midnight, binged an entire series. Should've studied instead.
        """
    },
    {
        "id": "j004",
        "date": "2024-01-18",
        "text": """
        Attended all classes today. Took detailed notes in the seminar. After college, went to 
        the library and did revision for 3 hours. Called mom in the evening, she's doing well.
        Cooked dinner at home - made pasta. Listened to music while cooking. Early to bed at 10pm.
        """
    },
    {
        "id": "j005",
        "date": "2024-01-19",
        "text": """
        Skipped breakfast, woke up too late. Had a horrible anxiety attack during class presentation.
        Felt overwhelmed and lonely. Talked to my therapist in the afternoon, helped a bit.
        Ate junk food for lunch - pizza and chips. Tried to study but couldn't concentrate.
        Scrolled through social media for hours. Feeling depressed.
        """
    },
    {
        "id": "j006",
        "date": "2024-01-20",
        "text": """
        Saturday! Slept in till noon. Did yoga for 30 minutes after waking up. Felt peaceful.
        Went shopping with friends at the mall. Bought new clothes. Had coffee at Starbucks.
        Came home and practiced guitar for an hour. Read a book in the evening. 
        Journaled my thoughts before sleeping. Good self-care day.
        """
    },
    {
        "id": "j007",
        "date": "2024-01-21",
        "text": """
        Went to church in the morning. Prayed and felt spiritually connected. Sunday brunch with 
        family was lovely. Played football with friends in the park. Got some exercise and fresh air.
        Worked on side project coding for 2 hours. Feeling accomplished and happy.
        """
    },
    {
        "id": "j008",
        "date": "2024-01-22",
        "text": """
        Back to the grind. Attended morning lecture but felt tired. Took a power nap in the afternoon.
        Finished my assignment and submitted it on time! Celebrated with friends by going to the movies.
        Watched a great film. Had dinner at a restaurant. Stayed hydrated all day, drank lots of water.
        """
    },
    {
        "id": "j009",
        "date": "2024-01-23",
        "text": """
        Started the day with meditation and stretching. Went to all my classes. Studied in the library 
        with my study group. We prepared for exams together. Skipped lunch because I was too busy.
        Worked till evening. Came home exhausted. Ordered fast food, ate McDonald's. 
        Played video games for 3 hours to relax. Valorant session with online friends.
        """
    },
    {
        "id": "j010",
        "date": "2024-01-24",
        "text": """
        Feeling burned out. Couldn't get out of bed. Skipped all classes. Stayed in room all day.
        Watched YouTube videos and TikTok. Wasted entire day on screen time. Eyes hurting from too much phone.
        Didn't eat proper meals, just snacks. Argued with roommate about cleaning. Feeling stressed and alone.
        Tried to study at night but gave up after 10 minutes.
        """
    },
    {
        "id": "j011",
        "date": "2024-01-25",
        "text": """
        Decided to turn things around. Woke up at 6am. Did morning exercise - went cycling for 40 minutes.
        Ate a nutritious breakfast with fruits and yogurt. Made a study plan and organized my tasks.
        Studied for 5 hours with proper breaks. Took breaks every hour. Practiced gratitude journaling.
        Listed three things I'm thankful for. Went for an evening walk. Early to bed at 9:30pm.
        """
    },
    {
        "id": "j012",
        "date": "2024-01-26",
        "text": """
        Exam day! Woke up stressed and anxious. Had coffee but skipped breakfast, stomach upset.
        Exam went okay, not great. After exam, met friends and we went out drinking. Had too many beers.
        Got drunk and had fun but now feeling guilty. Came home late at 2am. Bad decision before exam week.
        """
    },
    {
        "id": "j013",
        "date": "2024-01-27",
        "text": """
        Hangover. Felt terrible all morning. Drank water and took rest. Missed morning classes.
        Attended afternoon seminar, still feeling off. Studied a bit in evening. Called dad and 
        told him about exam stress. He gave good advice. Cooked a simple meal at home. 
        Cleaned my room and organized desk. Feeling better by night. Learned my lesson about drinking.
        """
    },
    {
        "id": "j014",
        "date": "2024-01-28",
        "text": """
        Weekend again. Went swimming at the pool in morning. Great cardio workout. 
        Came back and did laundry, finally! Worked on my creative project - painted for 2 hours.
        Art therapy is amazing. Listened to a podcast about productivity. Made healthy lunch.
        Read a book in the afternoon - fiction novel. Took photos at sunset, photography session.
        Peaceful and productive day. Feeling content and grateful.
        """
    },
    {
        "id": "j015",
        "date": "2024-01-29",
        "text": """
        Sunday study marathon. Started at 9am with revision. Took organized notes. 
        Had a video call with study partner, we practiced problems together. Break for lunch.
        Ate salad and grilled chicken. Back to studying. Did 8 hours total today!
        Exhausted but prepared. Quick meditation before bed. Tomorrow is the big exam. Nervous but ready.
        """
    },
    {
        "id": "j016",
        "date": "2024-01-30",
        "text": """
        Final exam done! It went well! Felt confident and happy. Celebrated by treating myself to 
        ice cream. Met friends at cafe, we all felt relieved. Went to a concert in evening - live music!
        Danced and had amazing time. Came home energized. Reflected on the whole exam week journey.
        Proud of my effort. Time to relax for a few days before next semester starts.
        """
    },
    {
        "id": "j017",
        "date": "2024-01-31",
        "text": """
        No alarms today! Slept till 11am. Lazy morning in bed scrolling phone. Brunch with family at home.
        Mom made my favorite food. Spent quality time with parents. Helped with gardening, watered plants.
        Went for nature walk in nearby park. Fresh air and greenery felt healing. 
        Came back and started planning goals for next month. Evening tea and cookies. Simple happy day.
        """
    },
    {
        "id": "j018",
        "date": "2024-02-01",
        "text": """
        New month, new energy! Went for early morning jog. Trying to build a running habit.
        Made breakfast smoothie with protein powder. Attended volunteer work at community center.
        Helped with teaching kids. Felt fulfilled doing something for others. 
        Came back and researched new online courses. Want to learn web development. 
        Practiced piano for 45 minutes. Musical skills improving slowly.
        """
    },
    {
        "id": "j019",
        "date": "2024-02-02",
        "text": """
        Mixed emotions today. Woke up feeling motivated but lost energy by afternoon. 
        Went to gym in morning, did strength training. Attended team meeting for group project.
        We have a lot of work ahead. Felt overwhelmed by deadlines. Had anxiety about time management.
        Skipped dinner, not hungry. Browsed news and current events. World feels heavy sometimes.
        Talked to friend on phone call, she understands. Feeling better after venting.
        """
    },
    {
        "id": "j020",
        "date": "2024-02-03",
        "text": """
        Saturday night out! Went on a date with my partner. Dinner at Italian restaurant was romantic.
        Walked by the riverside holding hands. Felt loved and connected. Came home and had quality time together.
        Watched a movie at home, romantic comedy. Made hot chocolate. Feeling happy and grateful for relationship.
        These moments matter most. Early sleep together, peaceful night.
        """
    }
]

def generate_test_data(output_path: str = "data/raw/test_journals.csv"):
    """Generate test dataset and save to CSV"""
    
    # Create DataFrame
    df = pd.DataFrame(SAMPLE_JOURNALS)
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Clean text (remove extra whitespace)
    df['text'] = df['text'].str.strip()
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Test dataset created: {output_path}")
    print(f"   - {len(df)} journal entries")
    print(f"   - Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   - Avg text length: {df['text'].str.len().mean():.0f} characters")
    
    # Also save as parquet
    parquet_path = output_file.with_suffix('.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f"✅ Also saved as: {parquet_path}")
    
    # Print sample
    print("\n" + "="*70)
    print("SAMPLE ENTRY:")
    print("="*70)
    print(f"ID: {df.iloc[0]['id']}")
    print(f"Date: {df.iloc[0]['date']}")
    print(f"Text: {df.iloc[0]['text'][:200]}...")
    print("="*70)
    
    return df


def generate_statistics(df: pd.DataFrame):
    """Print dataset statistics"""
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    # Basic stats
    print(f"Total entries: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Avg text length: {df['text'].str.len().mean():.0f} chars")
    print(f"Min text length: {df['text'].str.len().min()} chars")
    print(f"Max text length: {df['text'].str.len().max()} chars")
    
    # Word counts
    df['word_count'] = df['text'].str.split().str.len()
    print(f"\nAvg word count: {df['word_count'].mean():.0f} words")
    print(f"Min word count: {df['word_count'].min()} words")
    print(f"Max word count: {df['word_count'].max()} words")
    
    # Count mentions of seed habits (rough approximation)
    habits_mentioned = {
        'sleep': ['sleep', 'slept', 'sleeping', 'woke', 'wake'],
        'study': ['study', 'studied', 'studying', 'revision', 'exam'],
        'exercise': ['exercise', 'gym', 'workout', 'run', 'running', 'jog', 'yoga', 'cycling', 'swim'],
        'social_media': ['instagram', 'tiktok', 'facebook', 'scrolling', 'reels', 'phone'],
        'eat': ['breakfast', 'lunch', 'dinner', 'meal', 'ate', 'eating', 'food'],
        'anxiety': ['anxious', 'anxiety', 'stress', 'stressed', 'overwhelmed', 'worried'],
        'happy': ['happy', 'proud', 'grateful', 'content', 'joyful', 'excited'],
        'friends': ['friends', 'social', 'party', 'met', 'hang', 'celebrate']
    }
    
    print("\n" + "-"*70)
    print("HABIT MENTIONS (approximate):")
    print("-"*70)
    for habit, keywords in habits_mentioned.items():
        count = sum(df['text'].str.lower().str.contains('|'.join(keywords), regex=True))
        print(f"{habit:15s}: {count:2d} entries ({count/len(df)*100:.0f}%)")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Generate test data
    df = generate_test_data("data/raw/test_journals.csv")
    
    # Print statistics
    generate_statistics(df)
    
    print("🎉 Test dataset generation complete!")
    print("\nNext steps:")
    print("1. Run span extraction:")
    print("   python src/extraction/extract_regex.py \\")
    print("     --input data/raw/test_journals.csv \\")
    print("     --output results/spans/test_spans.parquet \\")
    print("     --seed-ontology seeds/seed_ontology.json")
    print("\n2. Run keyword mining:")
    print("   python src/extraction/keyword_mine.py \\")
    print("     --input data/raw/test_journals.csv \\")
    print("     --output results/spans/test_keywords.parquet \\")
    print("     --seed-ontology seeds/seed_ontology.json")