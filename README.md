# Simulador Causal de Política Pública

**Innovación Renovable y Transición Energética Baja en Carbono**

[![Streamlit App]((https://angelruiz-simulador-causal-politicaeinversion.streamlit.app/))


Herramienta interactiva de simulación de escenarios contrafactuales de política
pública derivada del artículo *"The Leader's Paradox in Clean Technology:
Catch-Up Returns to Public R&D and Green Crowding-Out Across Heterogeneous
Economies"* — tesis de maestría de Angel A. Ruiz Muñiz (Universidad
Autónoma de San Luis Potosí — SECIHTI).

## Motor causal

Coeficientes θ heterogéneos v11 (spec S3b) estimados sobre un panel balanceado
de 11 economías (Alemania, Brasil, Canadá, Chile, China, Corea del Sur,
Dinamarca, Estados Unidos, Francia, Japón, México) durante 2007–2024, mediante
un modelo híbrido de tres capas complementarias:

1. **Panel FE con errores estándar Driscoll–Kraay** (BW=4). Robusto a
   heteroscedasticidad, autocorrelación y dependencia seccional cruzada.
2. **Double/Debiased Machine Learning secuencial** con XGBoost como estimador
   *nuisance*, siguiendo Chernozhukov et al. (2018).
3. **Mediación por bootstrap paramétrico Monte Carlo** con N=10 000 réplicas,
   siguiendo Preacher & Hayes (2008).

## Funcionalidades

- Selector de los 11 países del panel de estudio.
- Cuatro palancas de política: GIDE, Crédito privado, IED y Pagos por uso de
  propiedad intelectual — con sliders de nivel sostenido y de impulso puntual.
- Horizonte de proyección de 1, 3, 5 o 10 años.
- Intervalos de confianza al 95 % vía bootstrap paramétrico (N=2 000).
- Descomposición Top Drivers y canal de transmisión GIDE → Patentes → Share LC.
- Comparativo simultáneo sobre los 11 países + Mapa de la Paradoja del Líder (H4).
- Análisis sexenal México (Calderón, Peña Nieto, AMLO, Sheinbaum) con
  escenarios contrafactuales y prospectivos anclados en marcos jurídicos
  verificables (LAERFTE 2008, Reforma Energética 2013–14, LTE 2015, Reforma
  LIE 2021, LGMHCTI 2023, LSE 2025 y Plan México 2025).
- Exportación a PDF de 3–4 páginas del escenario ejecutado.

## Deployment

Aplicación desplegada en **Streamlit Community Cloud** con hosting gratuito y
URL permanente. Para replicar el despliegue:

1. Fork o clone este repositorio.
2. Regístrate en [share.streamlit.io](https://share.streamlit.io) con GitHub.
3. **Deploy an app** → seleccionar este repositorio → *main branch* → archivo
   principal `streamlit_app.py`.
4. La URL pública queda operativa en 3–5 minutos.

Para correr localmente:

```bash
git clone <URL_DEL_REPO>
cd <NOMBRE_DEL_REPO>
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Referencias metodológicas verificables

- Abramovitz, M. (1986). Catching up, forging ahead, and falling behind.
  *The Journal of Economic History, 46*(2), 385–406.
  https://doi.org/10.1017/S0022050700046209
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
  Newey, W., & Robins, J. (2018). Double/debiased machine learning for
  treatment and structural parameters. *The Econometrics Journal, 21*(1),
  C1–C68. https://doi.org/10.1111/ectj.12097
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix
  estimation with spatially dependent panel data. *Review of Economics and
  Statistics, 80*(4), 549–560. https://doi.org/10.1162/003465398557825
- Fagerberg, J., & Srholec, M. (2008). National innovation systems,
  capabilities and economic development. *Research Policy, 37*(9), 1417–1435.
  https://doi.org/10.1016/j.respol.2008.06.003
- Preacher, K. J., & Hayes, A. F. (2008). Asymptotic and resampling strategies
  for assessing and comparing indirect effects in multiple mediator models.
  *Behavior Research Methods, 40*(3), 879–891.
  https://doi.org/10.3758/BRM.40.3.879
- Unruh, G. C. (2000). Understanding carbon lock-in. *Energy Policy, 28*(12),
  817–830. https://doi.org/10.1016/S0301-4215(00)00070-7

## Datos

El archivo `data.csv` contiene el panel consolidado 2007–2024 con las variables
del pipeline econométrico. Fuentes primarias: World Bank Open Data, OECD Main
Science and Technology Indicators, IRENA Renewable Energy Statistics, Our
World in Data, y bases nacionales complementarias.

## Autor

**Angel Adán Ruiz Muñiz**
Maestría en Ciudades Sostenibles — Universidad Autónoma de San Luis Potosí
Secretaría de Ciencia, Humanidades, Tecnología e Innovación (SECIHTI)

## Licencia

GNU Affero General Public License v3.0 (AGPLv3)** para fines académicos, de investigación y de código abierto — véase [LICENSE](LICENSE).
