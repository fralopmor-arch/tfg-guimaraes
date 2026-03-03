import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Datos nominales (ejemplo)
# -------------------------
P2 = 7.5e3          # Potencia mecánica nominal en el eje [W]
V_ll = 400          # Tensión línea-línea [V]
f = 50              # Frecuencia [Hz]
I1_nominal = 14.2   # Corriente nominal [A]
pf = 0.85           # Factor de potencia nominal
eta = 0.90          # Rendimiento nominal
nR = 1450           # Velocidad nominal [rpm]
pares_polos = 2     # Número de pares de polos

# Relaciones típicas de catálogo
rel_Iarr = 6.0      # I_arr / I_n
rel_Marr = 2.0      # M_arr / M_n
rel_Mmax = 2.5      # M_max / M_n


# ----------------------------------------
# Modelo físico simplificado por fase (Y)
# ----------------------------------------
def par_electromagnetico(s: np.ndarray | float, V1: float, r1: float, x1: float, r2p: float, x2p: float, ws: float) -> np.ndarray:
	"""Par electromagnético M(s) del motor de inducción.

	M(s) = 3 * V1^2 * (r2'/s) / ( ws * [ (r1 + r2'/s)^2 + (x1 + x2')^2 ] )
	"""
	s_arr = np.asarray(s, dtype=float)
	s_safe = np.clip(s_arr, 1e-5, None)

	r2_over_s = r2p / s_safe
	denom = ws * ((r1 + r2_over_s) ** 2 + (x1 + x2p) ** 2)
	return 3.0 * (V1 ** 2) * r2_over_s / denom


def corriente_estator(s: np.ndarray | float, V1: float, r1: float, x1: float, r2p: float, x2p: float) -> np.ndarray:
	"""Corriente por fase (rama de magnetización despreciada en el modelo simplificado)."""
	s_arr = np.asarray(s, dtype=float)
	s_safe = np.clip(s_arr, 1e-5, None)
	z_real = r1 + r2p / s_safe
	z_imag = x1 + x2p
	z_abs = np.sqrt(z_real ** 2 + z_imag ** 2)
	return V1 / z_abs


def error_relativo_pct(valor: float, referencia: float) -> float:
	return 100.0 * (valor - referencia) / referencia


def main() -> None:
	# Magnitudes síncronas
	ns = 60.0 * f / pares_polos                   # rpm
	ws = 2.0 * np.pi * ns / 60.0                  # rad/s
	sR = (ns - nR) / ns

	# Tensión de fase (conexión en estrella)
	V1 = V_ll / np.sqrt(3.0)

	# Estimación de parámetros equivalentes a partir de datos nominales
	Pin = P2 / eta
	r1 = (Pin - P2) / (3.0 * I1_nominal ** 2)

	I2_est = I1_nominal * pf
	r2p = (sR / (1.0 - sR)) * P2 / (3.0 * I2_est ** 2)

	s_k_from_ratio = sR * (rel_Marr + np.sqrt(rel_Marr ** 2 - 1.0))
	xeq = np.sqrt((r2p / s_k_from_ratio) ** 2 - r1 ** 2)
	x1 = xeq / 2.0
	x2p = xeq / 2.0

	# Referencias nominales (catálogo)
	wn = 2.0 * np.pi * nR / 60.0
	Mn = P2 / wn
	Marr_ref = rel_Marr * Mn
	Mmax_ref = rel_Mmax * Mn
	Iarr_ref = rel_Iarr * I1_nominal

	# Puntos del modelo
	MR = float(par_electromagnetico(sR, V1, r1, x1, r2p, x2p, ws))
	Marr = float(par_electromagnetico(1.0, V1, r1, x1, r2p, x2p, ws))
	s_max = r2p / np.sqrt(r1 ** 2 + (x1 + x2p) ** 2)
	Mmax = float(par_electromagnetico(s_max, V1, r1, x1, r2p, x2p, ws))
	Iarr = float(corriente_estator(1.0, V1, r1, x1, r2p, x2p))

	print("=== Parámetros del modelo (por fase) ===")
	print(f"r1  = {r1:.4f} ohm")
	print(f"r2' = {r2p:.4f} ohm")
	print(f"x1  = {x1:.4f} ohm")
	print(f"x2' = {x2p:.4f} ohm")
	print(f"ns  = {ns:.1f} rpm")
	print(f"ws  = {ws:.3f} rad/s")
	print()

	print("=== Verificación de puntos clave ===")
	print(f"M(sR) (modelo)      = {MR:.2f} N.m | Mn (ref)   = {Mn:.2f} N.m | error = {error_relativo_pct(MR, Mn):+.2f}%")
	print(f"M(1)  (modelo)      = {Marr:.2f} N.m | Marr (ref)= {Marr_ref:.2f} N.m | error = {error_relativo_pct(Marr, Marr_ref):+.2f}%")
	print(f"s_max (modelo)      = {s_max:.4f}")
	print(f"M(s_max) (modelo)   = {Mmax:.2f} N.m | Mmax (ref)= {Mmax_ref:.2f} N.m | error = {error_relativo_pct(Mmax, Mmax_ref):+.2f}%")
	print(f"I_arranque (modelo) = {Iarr:.2f} A   | Iarr (ref)= {Iarr_ref:.2f} A   | error = {error_relativo_pct(Iarr, Iarr_ref):+.2f}%")

	# Curva par-deslizamiento (0 a 1)
	s_vec = np.linspace(1e-4, 1.0, 1000)
	m_vec = par_electromagnetico(s_vec, V1, r1, x1, r2p, x2p, ws)

	plt.figure(figsize=(8, 5))
	plt.plot(s_vec, m_vec, linewidth=2, label="M(s) modelo")
	plt.scatter([sR, 1.0, s_max], [MR, Marr, Mmax], zorder=5, label="Puntos clave")
	plt.xlabel("Deslizamiento s")
	plt.ylabel("Par electromagnético M [N.m]")
	plt.title("Curva Par × Deslizamiento (Modelo de Guimarães)")
	plt.grid(True, alpha=0.3)
	plt.xlim(0.0, 1.0)
	plt.legend()
	plt.tight_layout()

	backend = plt.get_backend().lower()
	if "agg" in backend:
		output_path = "curva_par_deslizamiento.png"
		plt.savefig(output_path, dpi=150)
		print(f"Curva par-deslizamiento guardada en: {output_path}")
	else:
		plt.show()


if __name__ == "__main__":
	main()