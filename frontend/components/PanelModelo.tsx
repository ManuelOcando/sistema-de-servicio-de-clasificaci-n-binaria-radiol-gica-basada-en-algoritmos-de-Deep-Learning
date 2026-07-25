/**
 * Ficha de rendimiento del modelo desplegado.
 *
 * Son cifras medidas, no declarativas: proceden de evaluar el modelo sobre la
 * particion de prueba reservada (900 imagenes que no intervinieron en el
 * entrenamiento, verificado por hash exacto y perceptual). Se muestran en la
 * propia interfaz para que quien la usa pueda situar el resultado que acaba
 * de recibir.
 *
 * Cuatro cifras de cabecera se presentan como fila de indicadores, no como
 * grafico de barras: no hay comparacion entre categorias que hacer.
 */

const METRICAS = [
  { etiqueta: "Exactitud", valor: 0.9889, ayuda: "Aciertos sobre el total" },
  {
    etiqueta: "Sensibilidad",
    valor: 0.981,
    ayuda: "Anomalías detectadas sobre las existentes",
  },
  {
    etiqueta: "Especificidad",
    valor: 0.9958,
    ayuda: "Casos normales identificados correctamente",
  },
  { etiqueta: "AUC-ROC", valor: 0.9998, ayuda: "Capacidad de ordenar los casos" },
];

export default function PanelModelo() {
  return (
    <section className="rounded-xl border border-[var(--borde)] bg-[var(--panel)] p-5">
      <header className="mb-4">
        <h2 className="text-sm font-semibold">Rendimiento del modelo</h2>
        <p className="mt-1 text-xs text-[var(--tinta-tenue)]">
          Medido sobre 900 radiografías reservadas, ajenas al entrenamiento
        </p>
      </header>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-4">
        {METRICAS.map(({ etiqueta, valor, ayuda }) => (
          <div key={etiqueta}>
            <dt className="text-xs text-[var(--tinta-media)]" title={ayuda}>
              {etiqueta}
            </dt>
            <dd className="mt-0.5 text-xl font-semibold text-[var(--tinta)]">
              {(valor * 100).toFixed(2)}
              <span className="ml-0.5 text-xs font-normal text-[var(--tinta-tenue)]">%</span>
            </dd>
          </div>
        ))}
      </dl>

      <p className="mt-4 border-t border-[var(--borde)] pt-3 text-xs leading-relaxed text-[var(--tinta-tenue)]">
        Arquitectura YOLO11m-cls. Seleccionada frente a las variantes nano y
        xlarge: la mayor no aportó exactitud medible siendo 2,7 veces más pesada.
      </p>
    </section>
  );
}
