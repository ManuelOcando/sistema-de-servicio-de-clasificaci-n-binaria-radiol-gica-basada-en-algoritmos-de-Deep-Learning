"use client";

/**
 * Medidor de confianza frente al umbral de derivacion.
 *
 * La forma corresponde al trabajo del dato: una unica proporcion medida
 * contra un limite, que se representa con un medidor de pista continua, no
 * con un grafico de sectores.
 *
 * La escala va de 0 a 100 sin recortar. Ampliarla al tramo 90-100 haria mas
 * vistosa la aguja, pero exageraria diferencias irrelevantes y ocultaria lo
 * unico que importa aqui: si el valor queda por encima o por debajo del
 * umbral. Cuando el modelo se equivoca, su confianza cae de verdad (0,71 de
 * media en los errores medidos), y en esa escala se ve.
 */

const RADIO = 78;
const CENTRO = 100;
const ANGULO_INICIAL = 135;
const BARRIDO = 270;
const LONGITUD_ARCO = 2 * Math.PI * RADIO * (BARRIDO / 360);

type Props = {
  confianza: number;
  umbral: number;
};

function puntoEn(proporcion: number, radio: number) {
  const angulo = ((ANGULO_INICIAL + proporcion * BARRIDO) * Math.PI) / 180;
  return {
    x: CENTRO + radio * Math.cos(angulo),
    y: CENTRO + radio * Math.sin(angulo),
  };
}

export default function MedidorConfianza({ confianza, umbral }: Props) {
  const proporcion = Math.min(1, Math.max(0, confianza));
  const bajoUmbral = confianza < umbral;

  const inicioArco = puntoEn(0, RADIO);
  const finArco = puntoEn(1, RADIO);
  const marcaInterior = puntoEn(umbral, RADIO - 11);
  const marcaExterior = puntoEn(umbral, RADIO + 9);

  const trazoLleno = proporcion * LONGITUD_ARCO;

  return (
    <div className="flex flex-col items-center">
      <svg
        viewBox="0 0 200 176"
        className="w-full max-w-[15rem]"
        role="img"
        aria-label={`Confianza ${(confianza * 100).toFixed(2)} por ciento. Umbral de revisión en ${(umbral * 100).toFixed(0)} por ciento.`}
      >
        {/* Pista: paso tenue de la misma rampa que el relleno */}
        <path
          d={`M ${inicioArco.x} ${inicioArco.y} A ${RADIO} ${RADIO} 0 1 1 ${finArco.x} ${finArco.y}`}
          fill="none"
          stroke="var(--acento-pista)"
          strokeWidth="10"
          strokeLinecap="round"
        />

        {/* Relleno */}
        <path
          d={`M ${inicioArco.x} ${inicioArco.y} A ${RADIO} ${RADIO} 0 1 1 ${finArco.x} ${finArco.y}`}
          fill="none"
          stroke={bajoUmbral ? "var(--estado-aviso)" : "var(--acento)"}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${trazoLleno} ${LONGITUD_ARCO}`}
          style={{ transition: "stroke-dasharray 0.7s cubic-bezier(0.22,1,0.36,1)" }}
        />

        {/* Marca del umbral */}
        <line
          x1={marcaInterior.x}
          y1={marcaInterior.y}
          x2={marcaExterior.x}
          y2={marcaExterior.y}
          stroke="var(--tinta-media)"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <text
          x={marcaExterior.x}
          y={marcaExterior.y - 6}
          textAnchor="middle"
          className="fill-[var(--tinta-tenue)] text-[9px]"
        >
          {umbral}
        </text>

        {/* Cifra principal: tinta neutra, nunca el color del dato */}
        <text
          x={CENTRO}
          y={CENTRO + 4}
          textAnchor="middle"
          className="fill-[var(--tinta)] text-[34px] font-semibold"
        >
          {(confianza * 100).toFixed(1)}
          <tspan className="fill-[var(--tinta-media)] text-[16px]">%</tspan>
        </text>
        <text
          x={CENTRO}
          y={CENTRO + 26}
          textAnchor="middle"
          className="fill-[var(--tinta-tenue)] text-[10px] uppercase tracking-wider"
        >
          Confianza
        </text>

        {/* Extremos de la escala */}
        <text
          x={inicioArco.x - 4}
          y={inicioArco.y + 16}
          textAnchor="middle"
          className="fill-[var(--tinta-tenue)] text-[9px]"
        >
          0
        </text>
        <text
          x={finArco.x + 4}
          y={finArco.y + 16}
          textAnchor="middle"
          className="fill-[var(--tinta-tenue)] text-[9px]"
        >
          100
        </text>
      </svg>

      <p className="mt-1 text-center text-xs leading-relaxed text-[var(--tinta-media)]">
        {bajoUmbral ? (
          <>
            Por debajo del umbral de {umbral}:{" "}
            <strong className="font-medium text-[var(--tinta)]">
              se recomienda revisión humana
            </strong>
            .
          </>
        ) : (
          <>Por encima del umbral de derivación ({umbral}).</>
        )}
      </p>
    </div>
  );
}
