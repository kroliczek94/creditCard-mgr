from statistics import mean

import RF
import mysql.connector
from timeit import default_timer as timer


create_experiment_query = "INSERT INTO EXPERIMENTS(dataset) VALUES('ds1')"
fetch_experiment_id = "Select max(id) from experiments"
create_instance_query = "INSERT INTO INSTANCES(experiment_id, time, accuracy, prec, recall, specificity, auc, fscore, gmean, wtdAcc)"

iterations = 5

def GetExperimentId():
    cursor.execute(create_experiment_query)
    cursor.execute(fetch_experiment_id)
    return cursor.fetchone()[0]


def CreateConnection():
    global cnx, cursor
    cnx = mysql.connector.connect(user='root', password='',
                                  host='localhost',
                                  database='cakephp')
    return cnx.cursor()


if __name__ == "__main__":
    cursor = CreateConnection()
    exp_id = GetExperimentId()
    values = []
    for i in range (iterations):
        start = timer()


        end = timer()
        values.append(end - start)

    final_result = mean(sorted(values)[1:-1])
    print(final_result)
    cnx.commit()
    cnx.close()

    RF.run()

