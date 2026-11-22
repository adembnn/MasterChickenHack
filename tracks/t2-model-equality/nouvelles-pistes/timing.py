import json,datetime,numpy as np,sys,collections
t=lambda d: datetime.datetime.fromisoformat(d["timestamp"].replace("Z","")).timestamp()
for f in sys.argv[1:]:
  A=[json.loads(l) for l in open(f)]; A=[d for d in A if d["type"]=="answer"]
  A.sort(key=t)
  T=np.array([t(a) for a in A])
  print(f, "gaps pct", np.percentile(np.diff(T),[5,25,50,75,95,99]).round(2))
  lat=T[4:]-T[:-4]; ln=np.array([len(a["response"]) for a in A[:-4]])
  print(" lat pct",np.percentile(lat,[5,25,50,75,95]).round(1)," corr len",np.corrcoef(lat,ln)[0,1].round(3))
  per=collections.defaultdict(list)
  for i,a in enumerate(A[:-4]): per[a["endpoint"]].append(lat[i]/max(1,len(a["response"])))
