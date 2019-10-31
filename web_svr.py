import web
import entrypoint_factor
urls = (
    '/factor_importance/(.*)', 'factor_importance',
    '/lower/(.*)', 'lower'
)
app = web.application(urls, globals())

class factor_importance:
    def GET(self, dataSourceId):
        print("OK, get the request")
        entrypoint_factor.caculateImportance(dataSourceId)
        print("OK, complete the request")

if __name__ == "__main__":
    app.run()