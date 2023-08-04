from server.router import get_router

from . import api

router = get_router()

router.register(r'v1/start', api.ChatStartViewSet, basename='start_chat')
router.register(r'v1/ask', api.ChatAskViewSet, basename='ask_chat')
router.register(r'v1/reset', api.ChatResetViewSet, basename='reset_chat')
