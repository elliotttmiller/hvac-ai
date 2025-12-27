import asyncio
from starlette.requests import Request

async def main():
    async def rcv():
        return {'type':'http.request','body':b'','more_body':False}

    scope = {
        'type': 'http',
        'method': 'GET',
        'path': '/',
        'headers': [],
        'query_string': b'',
        'client': None,
        'server': None,
        'scheme': 'http',
    }

    req = Request(scope, rcv)
    attrs = sorted([a for a in dir(req) if not a.startswith('__')])
    print('attrs_sample=', attrs[:80])
    print('has_scope=', hasattr(req, 'scope'))
    print('has_receive=', hasattr(req, 'receive'))
    print('has__send=', hasattr(req, '_send'))
    print('has_send=', hasattr(req, 'send'))

asyncio.run(main())
