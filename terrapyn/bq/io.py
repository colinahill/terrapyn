from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def check_table_exists(table_id, client=None):
	if client is None:
		client = bigquery.Client()
	try:
		client.get_table(table_id)
		return True
	except NotFound:
		return False
