import sqlite3
import os

import numpy as np

from .nlp import normalize

domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']

dbs = None


def load_databases(path):
    # loading databases
    dbs_ = {}
    for domain in domains:
        db = os.path.join(path, f'{domain}-dbase.db')
        conn = sqlite3.connect(db)
        c = conn.cursor()
        dbs_[domain] = c
    global dbs
    dbs = dbs_


def oneHotVector(num, domain, vector):
    """Return number of available entities for particular domain."""
    number_of_options = 6
    if domain != 'train':
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
        elif num == 1:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num == 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num == 3:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num == 4:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num >= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])
    else:
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
        elif num <= 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num <= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num <= 10:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num <= 40:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        else:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])

    return vector


def queryResult(domain, turn):
    """Returns the list of entities for a given domain
    based on the annotation of the belief state"""
    # query the db
    sql_query = f"select * from {domain}"

    flag = True
    # print turn['metadata'][domain]['semi']
    for key, val in turn['metadata'][domain]['semi'].items():
        if val not in [
            "",
            "dont care",
            'not mentioned',
            "don't care",
            "dontcare",
            "do n't care",
        ]:
            val2 = val.replace("'", "''")
            if flag:
                sql_query += " where "
                # val2 = normalize(val2)
                # change query for trains
                if key == 'arriveBy':
                    sql_query += f" {key} < '{val2}'"
                elif key == 'leaveAt':
                    sql_query += f" {key} > '{val2}'"
                else:
                    sql_query += f" {key}='{val2}'"
                flag = False
            else:
                # val2 = normalize(val2)
                if key == 'arriveBy':
                    sql_query += f" and {key} < '{val2}'"
                elif key == 'leaveAt':
                    sql_query += f" and {key} > '{val2}'"
                else:
                    sql_query += f" and {key}='{val2}" + r"'"

    # try:  # "select * from attraction  where name = 'queens college'"
    # print sql_query
    # print domain
    num_entities = len(dbs[domain].execute(sql_query).fetchall())

    return num_entities


def queryResultVenues(domain, turn, real_belief=False):
    # query the db
    sql_query = f"select * from {domain}"

    if real_belief == True:
        items = turn.items()
    elif real_belief == 'tracking':
        for slot in turn[domain]:
            key = slot[0].split("-")[1]
            val = slot[0].split("-")[2]
            if key == "price range":
                key = "pricerange"
            elif key == "leave at":
                key = "leaveAt"
            elif key == "arrive by":
                key = "arriveBy"
            if val != "do n't care":
                if flag:
                    sql_query += " where "
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    if key == 'arriveBy':
                        sql_query += f"{key} < '{val2}'"
                    elif key == 'leaveAt':
                        sql_query += f"{key} > '{val2}'"
                    else:
                        sql_query += f" {key}='" + val2 + r"'"
                    flag = False
                else:
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    if key == 'arriveBy':
                        sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                    elif key == 'leaveAt':
                        sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

            try:  # "select * from attraction  where name = 'queens college'"
                return dbs[domain].execute(sql_query).fetchall()
            except:
                return []  # TODO test it
    else:
        items = turn['metadata'][domain]['semi'].items()

    flag = True
    for key, val in items:
        if val not in [
            "",
            "dontcare",
            'not mentioned',
            "don't care",
            "dont care",
            "do n't care",
        ]:
            val2 = val.replace("'", "''")
            val2 = normalize(val2)
            if flag:
                sql_query += " where "
                if key == 'arriveBy':
                    sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                elif key == 'leaveAt':
                    sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                else:
                    sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                flag = False
            else:
                if key == 'arriveBy':
                    sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                elif key == 'leaveAt':
                    sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                else:
                    sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

    try:  # "select * from attraction  where name = 'queens college'"
        return dbs[domain].execute(sql_query).fetchall()
    except:
        return []  # TODO test it
