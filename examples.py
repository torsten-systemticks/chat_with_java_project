QUERY_EXAMPLES = """ 
# Show me all classes that declare less than 3 public methods
MATCH (c:Class)-[:DECLARES_METHOD]->(m:Method)
WHERE m.accessModifier = "public"
WITH c, COUNT(m) AS methodCount
WHERE methodCount < 3
RETURN c.name

# Sort the packages by the number of types
MATCH (p:Package)-[:CONTAINS]->(t:Type)
WITH p, COUNT(t) AS typeCount
ORDER BY typeCount DESC
RETURN p.name, typeCount

# What classes do not have constructors
MATCH (c:Class)
WHERE NOT (c)-[:DECLARES_CONSTRUCTOR]->(:Constructor)
RETURN c.name

# Which classes contain the highest number of methods? Show the top 10
MATCH (c:Class)-[:DECLARES_METHOD]->(m:Method)
WITH c, COUNT(m) AS methodCount
ORDER BY methodCount DESC
LIMIT 10
RETURN c.name, methodCount

# What interfaces do not have an implementation
MATCH (i:Interface)
WHERE NOT (i)<-[:IMPLEMENTS]-(:Class)
RETURN i.name
"""