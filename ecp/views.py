
"""
Prediction engine API
"""
from django.conf import settings

from rest_framework import status
from rest_framework import permissions
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes


@api_view(['GET'])
@permission_classes((permissions.AllowAny,))
def predict(request):
    """
    Get predictions for a particular user.
    """

    features = ['xCycleCode', 'xFamilyNum', 'xFaze', 'xAmper', 'xRegionName_Roustaei',
                'xRegionName_Shahri', 'xBakhshCode_1', 'xBakhshCode_2', 'xBakhshCode_4',
                'xTimeControlCode_1', 'xTimeControlCode_2', 'xTimeControlCode_3',
                'xTariffOldCode_1010', 'xTariffOldCode_1011', 'xTariffOldCode_1110',
                'xTariffOldCode_1111', 'xTariffOldCode_1990', 'xTariffOldCode_2110',
                'xTariffOldCode_2210', 'xTariffOldCode_2310', 'xTariffOldCode_2410',
                'xTariffOldCode_2510', 'xTariffOldCode_2610', 'xTariffOldCode_2710',
                'xTariffOldCode_2990', 'xTariffOldCode_2992', 'xTariffOldCode_3110',
                'xTariffOldCode_3210', 'xTariffOldCode_3310', 'xTariffOldCode_3410',
                'xTariffOldCode_3520', 'xTariffOldCode_3540', 'xTariffOldCode_3740',
                'xTariffOldCode_3991', 'xTariffOldCode_4410', 'xTariffOldCode_4610',
                'xTariffOldCode_4990', 'xTariffOldCode_5110', 'xTariffOldCode_5990',
                'month']

    if request.method == 'GET':
        try:
            user_id = request.GET.get('user_id')
            days = int(request.GET.get('days'))
            month = int(request.GET.get('month'))

            medium_predicted = 0
            high_predicted = 0
            low_predicted = 0

            message = None

            if len(settings.DATASET.loc[settings.DATASET['xSubscriptionId_fk'].isin([user_id])]):
                df = settings.DATASET.loc[settings.DATASET['xSubscriptionId_fk'].isin([
                                                                                      user_id])]
                df = df.iloc[-1:]

                df['month'] = month

                if df['xUsageGroupName_Keshavarzi'].iloc[0].any():
                    low_predicted = settings.AGRICULTURE_MODEL_LOW.predict(df[features])[
                        0] * days
                    medium_predicted = settings.AGRICULTURE_MODEL_MEDIUM.predict(df[features])[
                        0] * days
                    high_predicted = settings.AGRICULTURE_MODEL_HIGH.predict(df[features])[
                        0] * days

                if df['xUsageGroupName_Khanegi'].iloc[0].any():
                    low_predicted = settings.HOME_MODEL_LOW.predict(df[features])[
                        0] * days
                    medium_predicted = settings.HOME_MODEL_MEDIUM.predict(df[features])[
                        0] * days
                    high_predicted = settings.HOME_MODEL_HIGH.predict(df[features])[
                        0] * days

                if df['xUsageGroupName_Omoomi'].iloc[0].any():
                    low_predicted = settings.PUBLIC_MODEL_LOW.predict(df[features])[
                        0] * days
                    medium_predicted = settings.PUBLIC_MODEL_MEDIUM.predict(df[features])[
                        0] * days
                    high_predicted = settings.PUBLIC_MODEL_HIGH.predict(df[features])[
                        0] * days

                if df['xUsageGroupName_Sanati'].iloc[0].any():
                    low_predicted = settings.INDUSTRIAL_MODEL_LOW.predict(df[features])[
                        0] * days
                    medium_predicted = settings.INDUSTRIAL_MODEL_MEDIUM.predict(df[features])[
                        0] * days
                    high_predicted = settings.INDUSTRIAL_MODEL_HIGH.predict(df[features])[
                        0] * days

                if df['xUsageGroupName_Sayer'].iloc[0].any():
                    low_predicted = settings.OTHER_MODEL_LOW.predict(df[features])[
                        0] * days
                    medium_predicted = settings.OTHER_MODEL_MEDIUM.predict(df[features])[
                        0] * days
                    high_predicted = settings.OTHER_MODEL_HIGH.predict(df[features])[
                        0] * days

            else:
                message = "This subscription ID is not valid."

            return Response({
                "success": True,
                "message": message,
                "prediction": {
                    "medium_predicted": round(medium_predicted, 10),
                    "high_predicted": round(high_predicted, 10),
                    "low_predicted": round(low_predicted, 10)}
            }, status=status.HTTP_200_OK)

        except Exception as ex:
            error_content = {
                "success": False,
                "message": "Hello! Error Happened.",
                "code": "PREDICTION_ERROR",
                "properties": {
                    "code": 1000,
                    "message": ex.__str__(),
                },
            }
            return Response(error_content, status=status.HTTP_404_NOT_FOUND)
