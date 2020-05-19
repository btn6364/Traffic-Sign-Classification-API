from flask import Flask, request, jsonify

app = Flask(__name__)
app.config["DEBUG"] = True

##################################
# ROUTE HANDLERS
##################################

@app.route("/", methods=["GET"])
def home():
    return "Welcome to home page!"

@app.errorhandler(404)
def page_not_found(err):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404


if __name__ == "__main__":
    app.run()
