import urllib.request, json

r = urllib.request.urlopen('http://127.0.0.1:8000/api/history?hours=6')
d = json.loads(r.read())
if 'greenhouse' in d:
    gh = d['greenhouse']
    print(f"Greenhouse: {gh['count']} records")
    print(f"  timestamps: {len(gh['timestamps'])}")
    print(f"  temperature: {len(gh['temperature'])}")
    print(f"  humidity: {len(gh['humidity'])}")
    if gh['temperature']:
        print(f"  temp range: {min(gh['temperature'])} ~ {max(gh['temperature'])}")
