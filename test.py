from memex import Memory

mem = Memory()

mem.save("User prefers Python")

print(mem.recall("favorite language"))
# User prefers Python