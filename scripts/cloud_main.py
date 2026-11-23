"""Development runner included in the Plotly Cloud package."""

from app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
