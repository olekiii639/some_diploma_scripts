
import numpy as np
import math
import z3
from scipy import stats

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
	['website2', 18500, 74]
]

metricsData = np.array(metricsData)

"""
...
Для расчета параметров регрессии построим расчетную таблицу
"""
x = metricsData[:, 1].astype(np.float64)
y = metricsData[:, 2].astype(np.float64)

ln_x = np.log(x)
y
ln_x_pow2 = ln_x ** 2
y_pow2 = y ** 2
ln_x * y
mul_ln_x_y = ln_x * y
params_calc_table = np.array([ln_x, y, ln_x_pow2, y_pow2, mul_ln_x_y])

# a·68 + b·∑(ln(x)) = ∑y
# a·∑(ln(x)) + b·∑(ln(x)^2) = ∑(ln(x)*y)

"""
...
Получаем эмпирические коэффициенты регрессии:
"""
sum_ln_x = np.sum(ln_x)
sum_y = np.sum(y)
sum_ln_x_pow2 = np.sum(ln_x_pow2)
sum_mul_ln_x_y = np.sum(mul_ln_x_y)
# len(metricsData) * a + b * sum_ln_x = sum_y
# sum_ln_x * a + b * sum_ln_x_pow2 = sum_mul_ln_x_y

# 68 * a + b * 722.340230 = 21355
# 722.340230 * a + b * 7996.298019 = 264724.357513

coefs_solution = np.linalg.solve(
	[[len(metricsData), sum_ln_x], [sum_ln_x, sum_ln_x_pow2]],
	[sum_y, sum_mul_ln_x_y]
)
a = coefs_solution[0] # [-931.15375975  117.22101603]
b = coefs_solution[1]

# Уравнение регрессии (эмпирическое уравнение регрессии):
# y = b * ln(x) - a

# 1. Параметры уравнения регрессии.
# Выборочные средние.
x_avg = sum_ln_x / len(metricsData)
y_avg = sum_y / len(metricsData)
xy_avg = sum_mul_ln_x_y / len(metricsData)

# Выборочные дисперсии:
S2x = sum_ln_x_pow2 / len(metricsData) - x_avg ** 2
S2y = np.sum(y_pow2) / len(metricsData) - y_avg ** 2

# Среднеквадратическое отклонение
Sx = math.sqrt(S2x)
Sy = math.sqrt(S2y)

# Коэффициент корреляции b можно находить по формуле, не решая систему непосредственно:
b_cor = (xy_avg - x_avg * y_avg) / S2x
a_cor = y_avg - b_cor * x_avg

# Бета – коэффициент
beta_coef = b_cor * Sx / Sy

# 2.1. Значимость коэффициента корреляции.
# вычислить наблюдаемое значение критерия (величина случайной ошибки)
t_nabl = beta_coef * math.sqrt(len(metricsData) - 2) / math.sqrt(1 - beta_coef ** 2)
t_critical = stats.t.ppf(1 - 0.125, len(metricsData) - 2)

# print(t_nabl)
# print(t_critical)

# 2.2. Интервальная оценка для коэффициента корреляции (доверительный интервал)
сonfidence_interval = [0, 0]
сonfidence_interval[0] = beta_coef - t_critical * \
						 math.sqrt((1 - beta_coef ** 2) / (len(metricsData) - 2))
сonfidence_interval[1] = beta_coef + t_critical * \
						 math.sqrt((1 - beta_coef ** 2) / (len(metricsData) - 2))

# print(сonfidence_interval)

# 1.3. Коэффициент эластичности
E = b_cor * x_avg / y_avg

# 1.4. Ошибка аппроксимации
A = np.sum(np.absolute((y - y_avg) / y)) / len(metricsData)
print(A)


# print("sum_ln_x = %f" % sum_ln_x)
# print("sum_y = %f" % sum_y)
# print("sum_ln_x_pow2 = %f" % sum_ln_x_pow2)
# print("sum_mul_ln_x_y = %f" % sum_mul_ln_x_y)

# sym_a = z3.Real('a')
# sym_b = z3.Real('b')
#
# coef_expression = z3.And(68 * sym_a + sym_b * sum_ln_x == sum_y,
# 						 sum_ln_x * sym_a + sym_b * sum_ln_x_pow2 == sum_mul_ln_x_y)
#
# print(coef_expression)
# z3.solve(coef_expression)

"""
a = -931.1538
b = 117.221
print(68 * a + b * sum_x)
print(sum_x * a + b * sum_ln_x_pow2)
"""





#print(params_calc_table[4])














