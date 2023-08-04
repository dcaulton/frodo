from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

class ChatStartViewSet(viewsets.ViewSet):
  def list(self, request):
    resp_data = {
      'status': 'started',
    }
    return Response(resp_data)


class ChatAskViewSet(viewsets.ViewSet):
  def list(self, request):
    resp_data = {
      'status': 'chatting',
    }
    return Response(resp_data)


class ChatResetViewSet(viewsets.ViewSet):
  def list(self, request):
    resp_data = {
      'status': 'reset',
    }
    return Response(resp_data)



