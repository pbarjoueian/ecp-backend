from django.conf import settings

from rest_framework import status
from rest_framework import permissions
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes

import joblib
import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostRegressor


@api_view(['GET'])
@permission_classes((permissions.AllowAny,))
def predict(request):
    """
    Get predictions for a particular user.
    """

    features = ['xCycleCode', 'xFamilyNum', 'xFaze', 'xAmper', 'xRegionName_Roustaei',
                'xRegionName_Shahri', 'xUsageGroupName_Keshavarzi', 'xUsageGroupName_Khanegi',
                'xUsageGroupName_Omoomi', 'xUsageGroupName_Sanati', 'xUsageGroupName_Sayer',
                'xBakhshCode_1', 'xBakhshCode_2', 'xBakhshCode_4',
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
                'days_difference', 'month']

    if request.method == 'GET':
        try:
            user_id = request.GET.get('user_id')
            days = int(request.GET.get('days'))
            month = int(request.GET.get('month'))
            
            medium_predicted = 0
            high_predicted = 0
            low_predicted = 0

            message = None

            if len(settings.ALL_CHANGE_IDS.loc[settings.ALL_CHANGE_IDS['xSubscriptionId_fk'].isin([user_id])]):
                df = settings.ALL_DF.loc[settings.ALL_DF['xSubscriptionId_fk'].isin([user_id])]
                df = df.iloc[-1:]
                df = df.drop(['xSubscriptionId_fk'], 1)
                df = df.drop(['xCounterBuldingNo'], 1)
                df = df.drop(df.columns[[0, 1]], 1).iloc[:, : 48]

                df['days_difference'] = days
                df['month'] = month

                if df['xTimeControlCode_1'].iloc[0].any():
                    medium_predicted = settings.ALL_MEDIUM_MODEL.predict(df[features])[
                        0] * days
                if df['xTimeControlCode_2'].iloc[0].any():
                    medium_predicted = settings.ALL_MEDIUM_MODEL.predict(df[features])[
                        0] * days
                    high_predicted = settings.ALL_HIGH_MODEL.predict(df[features])[
                        0] * days
                if df['xTimeControlCode_3'].iloc[0].any():
                    low_predicted = settings.ALL_LOW_MODEL.predict(df[features])[
                        0] * days
                    medium_predicted = settings.ALL_MEDIUM_MODEL.predict(df[features])[
                        0] * days
                    high_predicted = settings.ALL_HIGH_MODEL.predict(df[features])[
                        0] * days

                message = "Results may be not accurate due to changes in users data!"

            else:
                df = settings.NO_DF.loc[settings.NO_DF['xSubscriptionId_fk'].isin([user_id])]
                df = df.iloc[-1:]
                df = df.drop(['xSubscriptionId_fk'], 1)
                df = df.drop(['xCounterBuldingNo'], 1)
                df = df.drop(df.columns[[0, 1]], 1).iloc[:, : 48]

                df['days_difference'] = days
                df['month'] = month

                if df['xTimeControlCode_1'].any():
                    medium_predicted = settings.NO_MEDIUM_MODEL.predict(df[features])[
                        0] * days
                if df['xTimeControlCode_2'].any():
                    medium_predicted = settings.NO_MEDIUM_MODEL.predict(df[features])[
                        0] * days
                    high_predicted = settings.NO_HIGH_MODEL.predict(df[features])[
                        0] * days
                if df['xTimeControlCode_3'].any():
                    low_predicted = settings.NO_LOW_MODEL.predict(df[features])[
                        0] * days
                    medium_predicted = settings.NO_MEDIUM_MODEL.predict(df[features])[
                        0] * days
                    high_predicted = settings.NO_HIGH_MODEL.predict(df[features])[
                        0] * days

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
