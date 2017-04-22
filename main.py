import numpy as np
import cv2

from flask import Flask
from flask import jsonify
from flask import request
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker,relationship,scoped_session
from flask_sqlalchemy_session import flask_scoped_session
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, Float, DateTime
from dateutil import parser

from ripeness_predictor import is_ripe

app = Flask(__name__)
Base = declarative_base()


class TamatarData(Base):
    __tablename__ = 'tamatar'
    id = Column(Integer, primary_key=True)
    lat = Column(Float)
    lng = Column(Float)
    timestamp = Column(DateTime)
    ripe = Column(Integer)

def get_sqlite3_session(path):
    engine = create_engine('sqlite:///' + path)
    Base.metadata.create_all(engine)
    return scoped_session(sessionmaker(bind=engine))

session = flask_scoped_session(get_sqlite3_session("db.db"), app)


@app.route("/add", methods=["POST"])
def add_image():
    lat = float(request.form['lat'])
    lng = float(request.form['lng'])
    dt = parser.parse(request.form["date"])
    img_str = request.form['image']
    nparr = np.fromstring(img_str, np.uint8)
    img_np = cv2.imdecode(nparr, 1) # cv2.IMREAD_COLOR in OpenCV 3.1

    ripe = is_ripe(img_np)

    t = TamatarData(lat=lat, lng=lng, ripe=ripe)
    session.add(t)
    session.commit()
    return jsonify(status=True)


@app.route("/get")
def get_all():
    query = session.query(TamatarData)
    if 'from' in request.form:
        frm = parser.parse(request.form['from'])
        query = query.filter(TamatarData.timestamp >= frm)
    if 'to' in request.form:
        to = parser.parse(request.form['to'])
        query = query.filter(TamatarData.timestamp <= to)

    res = list(query)
    return jsonify(result=res)


app.run(host='0.0.0.0', threaded=True)
