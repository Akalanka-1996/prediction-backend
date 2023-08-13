from flask import current_app as app
import random
import string
import jwt
import datetime
from dotenv import load_dotenv


# load_dotenv()

# app = Flask(__name__)
# app.config["MONGO_URI"] = os.environ.get("MONGO_URI")
# app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")
# mongo = PyMongo(app)


def generate_random_user_id():
    random_suffix = "".join(random.choices(string.digits, k=6))
    user_id = f"USER{random_suffix}"
    return user_id

def create_jwt(user_id):
    payload = {
        "user_id": user_id,
        "exp": datetime.datetime.utcnow()
        + datetime.timedelta(days=1),  # Token expiration time
    }
    token = jwt.encode(payload, app.config["SECRET_KEY"], algorithm="HS256")
    return token

def verify_jwt(token):
    try:
        payload = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return None
