from insertrecord import insert_record
from datetime import datetime
import time

# Data class/structure
data = {
    'question': 'q1',
    'answer': 'an1',
    'date_enter': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    # Compute the time elapsed since start time and additional 2 seconds
    'time_elape': round((time.time() - datetime.now().timestamp()) + 2), # Adjusted to match the context
    'temperature': 0.7,
    'topp': 0.7,
    'topk': 0.7,
    'model_name': 'example_model_name',
    'local_rag': 'stemcell',
    'con_score': 0.7,
    'misc': 'example_misc_text'
}

# Insert record into the database
insert_record(data)


