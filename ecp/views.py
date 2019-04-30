from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework import permissions
from rest_framework.response import Response

import glob
from sklearn.externals import joblib
import os, fnmatch


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


@api_view(['GET'])
@permission_classes((permissions.AllowAny,))
def predict(request):
    """
    Get predictions for a particular user.
    """
    if request.method == 'GET':
        try:
            base_path = 'C:/Users/pbarjoueian/Documents/projects/repos/ecp/dataset/users_data/'
            user_id = request.GET.get('user_id')
            days = int(request.GET.get('days'))
            month = int(request.GET.get('month'))
            models = glob.glob(base_path + str(user_id) + '*.pkl')

            code = models[0].split("-")[2].split(".")[0]
            first_digit_code = (int(str(code)[:1]))
            last_digit_code = (int(str(code)[-1:]))

            medium_predicted = 0
            high_predicted = 0
            low_predicted = 0

            message = None

            if first_digit_code == 1:
                if last_digit_code == 0:
                    est = joblib.load(base_path + '\{0}-mediumDailyUsage-100.pkl'.format(str(user_id)))
                    medium_predicted = days * est.predict([[days, month]])[0]
                else:
                    est = joblib.load(base_path + '\{0}-mediumDailyUsage-101.pkl'.format(str(user_id)))
                    medium_predicted = days * est.predict([[days, month]])[0]
                    message = "Results may not be percise due to low training data!"
            elif first_digit_code == 2:
                if last_digit_code == 0:
                    est_medium = joblib.load(base_path + '\{0}-mediumDailyUsage-200.pkl'.format(str(user_id)))
                    medium_predicted = days * est_medium.predict([[days, month]])[0]
                    est_high = joblib.load(base_path + '\{0}-highDailyUsage-210.pkl'.format(str(user_id)))
                    high_predicted = days * est_high.predict([[days, month]])[0]
                else:
                    est_medium = joblib.load(base_path + '\{0}-mediumDailyUsage-201.pkl'.format(str(user_id)))
                    medium_predicted = days * est_medium.predict([[days, month]])[0]
                    est_high = joblib.load(base_path + '\{0}-highDailyUsage-211.pkl'.format(str(user_id)))
                    high_predicted = days * est_high.predict([[days, month]])[0]
                    message = "Results may not be percise due to low training data!"
            elif first_digit_code == 3:
                if last_digit_code == 0:
                    est_medium = joblib.load(base_path + '\{0}-mediumDailyUsage-300.pkl'.format(str(user_id)))
                    medium_predicted = days * est_medium.predict([[days, month]])[0]
                    est_high = joblib.load(base_path + '\{0}-highDailyUsage-310.pkl'.format(str(user_id)))
                    high_predicted = days * est_high.predict([[days, month]])[0]
                    est_low = joblib.load(base_path + '\{0}-lowDailyUsage-320.pkl'.format(str(user_id)))
                    low_predicted = days * est_low.predict([[days, month]])[0]

                else:
                    est_medium = joblib.load(base_path + '\{0}-mediumDailyUsage-301.pkl'.format(str(user_id)))
                    medium_predicted = days * est_medium.predict([[days, month]])[0]
                    est_high = joblib.load(base_path + '\{0}-mediumDailyUsage-311.pkl'.format(str(user_id)))
                    high_predicted = days * est_high.predict([[days, month]])[0]
                    est_low = joblib.load(base_path + '\{0}-mediumDailyUsage-321.pkl'.format(str(user_id)))
                    low_predicted = days * est_low.predict([[days, month]])[0]
                    message = "Results may not be percise due to low training data!"
            return Response(
                {"success": True,
                 "message": message,
                 "prediction": {"medium_predicted": medium_predicted,
                                "high_predicted": high_predicted,
                                "low_predicted": low_predicted}
                },
                status=status.HTTP_200_OK)

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
            return Response(
                error_content, status=status.HTTP_400_BAD_REQUEST)
