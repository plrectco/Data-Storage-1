#coding=utf-8
import MySQLdb
import csv
import numpy as np
conn= MySQLdb.connect(
        host='localhost',
        port = 3306,
        user='root',
        passwd='123',
        db ='iris',
        )

cur = conn.cursor()
try:
	#create the table 
	cur.execute("""create table flower
	(  
		sepal_length float not null, 
		sepal_width float not null, 
		petal_length float not null, 
		petal_width float not null, 
		category char(20) null default "-"
	);""");

	# insert data into the table
	# DECIMAL(10,6) is exactly type float
	csv_data = csv.reader(file('iris.csv'))
	for row in csv_data:
		cur.execute("INSERT INTO flower(sepal_length, sepal_width,"\
			" petal_length, petal_width, category )" \
			"VALUES( CAST(%s AS DECIMAL(10,6)), CAST(%s AS DECIMAL(10,6)),"\
			" CAST(%s AS DECIMAL(10,6)), CAST(%s AS DECIMAL(10,6)), %s)",row)
	cur.close()
	conn.commit()
except:
	conn.rollback()
#close the connection

conn.close()

# cur.execute("INSERT INTO flower(sepal_length, sepal_width,"\
# 	"petal_length, petal_width, category)"\
# 	"VALUES( 5.1, 3.5, 1.4, 0.2, 'Iris-setosa')")
