from memex import Memory

mem = Memory()

mem.save("User prefers Python")

print(mem.recall("favorite language"))
print(mem.recall("what does she prefer"))
# User prefers Python