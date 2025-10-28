from neo4j import GraphDatabase

class Neo4jConnector:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        Initialize connection to Neo4j Aura or local Neo4j.
        """
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))

    def close(self):
        """
        Close the Neo4j connection.
        """
        if self._driver:
            self._driver.close()

    def test_connection(self) -> str:
        """
        Test the connection by returning Neo4j version.
        """
        with self._driver.session(database=self._database) as session:
            result = session.run("RETURN 'Connected to Neo4j' AS output")
            return result.single()["output"]

    @property
    def driver(self):
        return self._driver

    @property
    def database(self):
        return self._database
