from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from starlette.responses import HTMLResponse
from typing import Optional

from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import HouseData, HousePricePredictor
from src.pipline.training_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HouseForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.Area: Optional[float] = None
        self.BHK: Optional[int] = None
        self.Bathroom: Optional[int] = None
        self.Furnishing: Optional[str] = None
        self.Locality: Optional[str] = None
        self.Parking: Optional[int] = None
        self.Status: Optional[str] = None
        self.Transaction: Optional[str] = None
        self.Type: Optional[str] = None

    async def get_house_data(self):
        form = await self.request.form()
        self.Area = float(form.get("Area"))
        self.BHK = int(form.get("BHK"))
        self.Bathroom = int(form.get("Bathroom"))
        self.Furnishing = form.get("Furnishing")
        self.Locality = form.get("Locality")
        self.Parking = int(form.get("Parking"))
        self.Status = form.get("Status")
        self.Transaction = form.get("Transaction")
        self.Type = form.get("Type")


@app.get("/", tags=["UI"])
async def index(request: Request):
    return templates.TemplateResponse("housedata.html", {"request": request, "context": "Rendering"})

@app.get("/train", tags=["Training"])
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Model training successful.")
    except Exception as e:
        return Response(f"Training failed. Error: {e}")

@app.post("/", tags=["Prediction"])
async def predictRouteClient(request: Request):
    try:
        form = HouseForm(request)
        await form.get_house_data()

        house_data = HouseData(
            Area=form.Area,
            BHK=form.BHK,
            Bathroom=form.Bathroom,
            Furnishing=form.Furnishing,
            Locality=form.Locality,
            Parking=form.Parking,
            Status=form.Status,
            Transaction=form.Transaction,
            Type=form.Type
        )

        input_df = house_data.to_dataframe()
        predictor = HousePricePredictor()
        prediction = predictor.predict(input_df)[0]

        return templates.TemplateResponse(
            "housedata.html",
            {"request": request, "context": f"Predicted Price: ₹{prediction:,.2f}"},
        )
    except Exception as e:
        return {"status": False, "error": str(e)}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
