安装依赖
```bash
python3 -m pip install -r requirements.txt
```

调用沙盒
```python
from kimina_client import KiminaSandboxClient, Snippet
server_heartbeat_record_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangjianfei09/cfs/kls_heartbeat"
server_timeout = 600
client = KiminaSandboxClient(
    heartbeat_record_path=server_heartbeat_record_path,
    http_timeout=server_timeout
)
proofs = [] # 所有待验证theorem/proof的list
snips = [Snippet(id=f"{idx}", code=str(proof)) for idx, proof in enumerate(proofs)]
response = client.check(snips=snips, max_workers=32, batch_size=1)
# 等待返回结果
results = sorted(response.results, key=lambda result: int(result.id))
statuses = [result.analyze().status for result in results]
# status说明
status_to_score = {
    "valid": 1.0,
    "sorry": 0.0,
    "lean_error": 0.0,  # Error in snippet, clearly identified by message of severity "error"
    "repl_error": 0.0,  # Error while running snippet, at REPL level
    "timeout_error": 0.0,   # Error caught at server level, which contains "timed out" in the error message
    "server_error": 0.0,    # Error caught at server level, which is not a timeout error
}
```