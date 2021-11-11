import numpy as np
import pandas as pd
from sklearn import *
import matplotlib.pyplot as plt

metricsData = \
    [
        ['easyappointments', 125700, 235],
        ['InvoicePlane', 69900, 275],
        ['grocery-crud', 217700, 45],
        ['Ecommerce-CodeIgniter-Bootstrap', 258600, 448],
        ['HookPHP', 83700, 141],
        ['ci-phpunit-test', 150500, 724],
        ['mapos', 163800, 253],
        ['cron-manager', 5200, 55],
        ['ci_bootstrap_3', 326300, 818],
        ['CodeIgniter-Aauth', 5500, 7],
        ['codeigniter-composer-installer', 766, 2],
        ['Mini-Inventory-and-Sales-Management-System', 98200, 198],
        ['Admin-Panel-User-Management-using-CodeIgniter', 383200, 187],
        ['CI-AdminLTE', 88400, 205],
        ['cms', 297900, 1606],
        ['codeigniter-phpmailer', 1900, 2],
        ['learnify', 1500000, 1245],
        ['vue-questionnaire', 190100, 378],
        ['codeigniter-phpstorm', 147, 2],
        ['datatables', 117, 24],
        ['Ignition-Go', 226200, 403],
        ['starter-public-edition-4', 1200000, 2409],
        ['identity-card', 26500, 3],
        ['CodeIgniter-develbar', 3100, 22],
        ['memcached-library', 531, 3],
        ['codeigniter-ss-twig', 3200, 6],
        ['Bonfire', 264900, 619],
        ['codeigniter-hmvc-modules', 930, 5],
        ['CodeIgniter-HMVC', 231100, 525],
        ['Codeigniter-login-logout-register', 78200, 156],
        ['codeigniter-angularjs-app', 11000, 168],
        ['Codeigniter-Crud-With-Angular-Js', 69200, 153],
        ['clocks', 109800, 264],
        ['Hoosk', 216500, 296],
        ['CodeIgniter-JWT-Sample', 25000, 707],
        ['luthier-ci', 21500, 64],
        ['Startblog', 120700, 220],
        ['playground', 13100, 63],
        ['classroombookings', 33000, 211],
        ['manga-tracker', 46300, 313],
        ['ci-markdown', 4200, 4],
        ['gitblog', 110900, 379],
        ['hr-payroll', 213700, 262],
        ['Simple-realtime-message-SocketIO-NodeJS-CI', 211400, 154],
        ['CodeIgniter-Websocket-Apache-Secure', 170500, 674],
        ['CodeIgniter-Ratchet-Websocket', 135900, 524],
        ['socialigniter', 31300, 254],
        ['bancha', 101800, 253],
        ['codeigniter_bootstrap_form_builder', 1700, 4],
        ['codeigniter4-user-guide', 29600, 77],
        ['InventorySystem', 800900, 716],
        ['skeleton', 282300, 338],
        ['ci4-album', 3300, 41],
        ['CodeIgniter3-online-shop', 1500000, 1691],
        ['cat', 185800, 318],
        ['flexicms', 70700, 699],
        ['ci-rest-jwt', 87800, 190],
        ['phpdotenv-for-codeigniter', 639, 3],
        ['CodeIgniter-API-Controller', 70800, 158],
        ['Mahana-Messaging-library-for-CodeIgniter', 1200, 2],
        ['SimpleRegistration', 72100, 175],
        ['user-registration-codeigniter', 69500, 160],
        ['Codeigniter-Social-Register', 2600, 6],
        ['Codeigniter-register-login-logout', 70700, 156],
        ['codeigniter4-user-authentication', 63500, 79],
        ['userreg', 152900, 338],
        ['User-Login-Registration-Codeigniter', 211100, 166],
        ['website2', 18500, 74],

        ['pp00pp0', 115750, 275],
        ['[InvoicePlane]', 61900, 235],
        ['[grocery-crud]', 117700, 25],
        ['[Ecommerce-CodeIgniter-Bootstrap]', 299950, 768],
        ['[HookPHP]', 83750, 142],
        ['[ci-phpunit-test]', 115500, 744],
        [']', 112850, 203],
        ['[cron-manager]', 3200, 35],
        ['[ci_bootstrap_3]', 26300, 118],
        ['[CodeIgniter-Aauth]', 500, 2],
        ['[codeigniter-composer-installer]', 769, 2],
        ['[Mini-Inventory-and-Sales-Management-System]', 88205, 188],
        ['[Admin-Panel-User-Management-using-CodeIgniter]', 353250, 157],
        ['[CI-AdminLTE]', 83400, 108],
        ['[cms]', 278906, 1316],
        ['[codeigniter-phpmailer]', 190, 1],
        ['[lear0noi[fy]', 1130000, 1123],
        ['[vue-questionnaire]', 115100, 279],
        ['[codeigniter-phpstorm]', 198, 3],
        ['[datatables]', 566, 3],

    ]

metricsData = np.array(metricsData)

"""
...
Для расчета параметров регрессии построим расчетную таблицу
"""
x = metricsData[:, 1].astype(np.float64)
y = metricsData[:, 2].astype(np.float64)

xy_arr = np.column_stack((x, y))

# Create Data - with Anomaly - as before.
# d1 = np.random.multivariate_normal(mean=np.array([-.5, 0]),
#                                   cov=np.array([[1, 0], [0, 1]]), size=100)
# print(d1)
# outliers = np.array([[0, 10], [0, 9.5]])
d = pd.DataFrame(np.concatenate([xy_arr], axis=0), columns=['Lines of code', 'Classes'])
###### Fit Elliptic Envelope ##############
contamination = 0.4  # We can set any value here as we will now use our own threshold
el = covariance.EllipticEnvelope(store_precision=True, assume_centered=False, support_fraction=None,
                                 contamination=contamination, random_state=0)
# Fit the data
el.fit(d)
############# New Part ################
# Create column that measures Mahalanobis distance
d['Mahalanobis Distance'] = el.mahalanobis(d)

mahalanobis_distance_arr = []
xy_arr_new = []
for i in range(0, len(metricsData)):
    distance = d['Mahalanobis Distance'][i]
    if distance < 0.85:
        xy_arr_new.append(xy_arr[i])
    else:
        print(f"discarding {xy_arr[i]}, Mahalanobis Distance {distance}")
    mahalanobis_distance_arr.append([xy_arr[i], distance])

# Create scatterplot and color the anomalies differently
plt.figure(figsize=(12, 6))
ax = plt.scatter(d['Lines of code'], d['Classes'], c=d['Mahalanobis Distance'], cmap='coolwarm')
# plt.title('Contamination = Does not matter for this method', weight = 'bold')
# ax = sns.scatterplot(d['Var 1'], d['Var 2'], c = d['Anomaly or Not'])
plt.xlabel('Lines of code')
plt.ylabel('Classes')
plt.colorbar(label='Mahalanobis Distance')
plt.grid()
plt.show()

# print(mahalanobis_distance_arr)










