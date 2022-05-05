class TestMiddleware(objext):
	def process_request(self, request):
		return "Hello"