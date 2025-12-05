#!/usr/bin/env python3
"""
FastAPI-based async gateway for multiple vLLM servers
High-performance proxy with streaming support and auto-documentation
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import asyncio
import uvicorn



app = FastAPI(
    title="vLLM Multi-Model Gateway",
    description="High-performance async gateway for multiple vLLM instances",
    version="2.0.0"
)

# Global state
endpoints: Dict[str, str] = {}
timeout = httpx.Timeout(300.0, connect=10.0)


def load_endpoints(label: str = 'vllm') -> Dict[str, str]:
    """Load endpoint mapping from server info file"""
    info_file = f"{label}_servers.json"
    if not Path(info_file).exists():
        raise FileNotFoundError(f"{info_file} not found. Run run_vllm_server.py first.")
    
    with open(info_file) as f:
        server_info = json.load(f)
    
    mapping = {info['model_name']: f"http://127.0.0.1:{info['port']}" 
               for info in server_info}
    
    print(f"Loaded {len(mapping)} model endpoints:")
    for name, url in mapping.items():
        print(f"  {name}: {url}")
    
    return mapping


async def proxy_stream(response: httpx.Response):
    """Stream response chunks asynchronously"""
    async for chunk in response.aiter_bytes():
        yield chunk


@app.get("/")
async def root():
    """Root endpoint with gateway info"""
    return {
        "service": "vLLM Multi-Model Gateway",
        "version": "2.0.0",
        "models": list(endpoints.keys()),
        "endpoints": {
            "models": "/v1/models",
            "chat": "/v1/chat/completions",
            "completions": "/v1/completions",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/v1/models")
async def list_models():
    """Merge model lists from all vLLM instances"""
    merged_data = {'object': 'list', 'data': []}
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            client.get(f"{endpoint}/v1/models")
            for endpoint in endpoints.values()
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for response in responses:
            if isinstance(response, httpx.Response) and response.status_code == 200:
                data = response.json()
                merged_data['data'].extend(data.get('data', []))
    
    return merged_data


@app.post("/v1/{path:path}")
async def proxy_post(path: str, request: Request):
    """Proxy POST requests to appropriate vLLM instance"""
    try:
        body = await request.json()
    except:
        raise HTTPException(400, "Invalid JSON body")
    
    model_name = body.get('model')
    if not model_name:
        raise HTTPException(400, "Missing 'model' field in request")
    
    if model_name not in endpoints:
        raise HTTPException(
            404, 
            detail=f"Model '{model_name}' not found. Available: {list(endpoints.keys())}"
        )
    
    target_url = f"{endpoints[model_name]}/v1/{path}"
    is_stream = body.get('stream', False)
    
    # Forward headers (exclude hop-by-hop)
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ['host', 'content-length', 'connection', 'transfer-encoding']
    }
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            if is_stream:
                # Streaming response
                response = await client.post(
                    target_url, 
                    json=body, 
                    headers=headers,
                    timeout=httpx.Timeout(300.0, connect=10.0)
                )
                return StreamingResponse(
                    proxy_stream(response),
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type='text/event-stream'
                )
            else:
                # Regular response
                response = await client.post(target_url, json=body, headers=headers)
                return JSONResponse(
                    content=response.json(),
                    status_code=response.status_code
                )
        except httpx.TimeoutException:
            raise HTTPException(504, "Backend server timeout")
        except httpx.RequestError as e:
            raise HTTPException(502, f"Backend connection error: {str(e)}")


@app.get("/health")
async def health_check():
    """Check health of all backend vLLM instances"""
    status = {}
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        tasks = {
            name: client.get(f"{endpoint}/health")
            for name, endpoint in endpoints.items()
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        for name, result in zip(tasks.keys(), results):
            if isinstance(result, httpx.Response):
                status[name] = {
                    'status': 'healthy' if result.status_code == 200 else 'unhealthy',
                    'status_code': result.status_code
                }
            else:
                status[name] = {
                    'status': 'unreachable',
                    'error': str(result)
                }
    
    all_healthy = all(s['status'] == 'healthy' for s in status.values())
    return JSONResponse(
        content={'models': status, 'overall': 'healthy' if all_healthy else 'degraded'},
        status_code=200 if all_healthy else 503
    )


@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint"""
    return {
        "registered_models": len(endpoints),
        "model_names": list(endpoints.keys())
    }


def main():
    parser = argparse.ArgumentParser(description='FastAPI vLLM Gateway')
    parser.add_argument('--port', type=int, default=8000, help='Gateway port')
    parser.add_argument('--label', type=str, default='vllm', help='vLLM servers label')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Gateway host')
    parser.add_argument('--workers', type=int, default=1, help='Uvicorn workers')
    args = parser.parse_args()
    
    global endpoints
    try:
        endpoints = load_endpoints(args.label)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print(f"\n🚀 Starting FastAPI gateway on {args.host}:{args.port}")
    print(f"📚 API docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )


if __name__ == '__main__':
    main()