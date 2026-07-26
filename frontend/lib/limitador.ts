/**
 * Limitador de peticiones por dirección IP.
 *
 * El servicio de inferencia se factura por uso y el saldo disponible es
 * reducido. Como la ruta intermediaria queda expuesta públicamente al
 * desplegar el frontend, cualquiera podría automatizar peticiones y agotarlo
 * en minutos. Este limitador es la contención mínima frente a ese escenario.
 *
 * Limitación conocida: el estado vive en memoria del proceso. Las funciones
 * serverless se reparten entre varias instancias, de modo que el límite
 * efectivo puede ser un múltiplo del configurado. Contiene el abuso casual y
 * los scripts simples, que es el riesgo real aquí; frenar a un atacante
 * decidido exigiría un almacén compartido, lo que implica contratar un
 * servicio externo.
 */

type Registro = { marcas: number[] };

const VENTANA_CORTA_MS = 60_000;
const MAXIMO_CORTO = 8;

const VENTANA_LARGA_MS = 60 * 60_000;
const MAXIMO_LARGO = 40;

// Se poda para que el mapa no crezca sin límite en una instancia longeva.
const MAXIMO_CLIENTES = 5_000;

const registros = new Map<string, Registro>();

export type Veredicto =
  | { permitido: true }
  | { permitido: false; esperarSegundos: number; motivo: string };

export function identificar(cabeceras: Headers): string {
  // En Vercel la IP real llega en x-forwarded-for; el primer valor es el
  // cliente y los siguientes son los proxies intermedios.
  const reenviado = cabeceras.get("x-forwarded-for");
  if (reenviado) return reenviado.split(",")[0].trim();
  return cabeceras.get("x-real-ip") ?? "desconocido";
}

function podar(ahora: number) {
  if (registros.size <= MAXIMO_CLIENTES) return;
  for (const [clave, registro] of registros) {
    if (registro.marcas.every((m) => ahora - m > VENTANA_LARGA_MS)) {
      registros.delete(clave);
    }
    if (registros.size <= MAXIMO_CLIENTES) break;
  }
}

export function permitir(cliente: string): Veredicto {
  const ahora = Date.now();
  const registro = registros.get(cliente) ?? { marcas: [] };

  const vigentes = registro.marcas.filter((m) => ahora - m < VENTANA_LARGA_MS);
  const recientes = vigentes.filter((m) => ahora - m < VENTANA_CORTA_MS);

  if (recientes.length >= MAXIMO_CORTO) {
    const espera = Math.ceil((VENTANA_CORTA_MS - (ahora - recientes[0])) / 1000);
    return {
      permitido: false,
      esperarSegundos: Math.max(1, espera),
      motivo: `Has alcanzado el límite de ${MAXIMO_CORTO} análisis por minuto.`,
    };
  }

  if (vigentes.length >= MAXIMO_LARGO) {
    const espera = Math.ceil((VENTANA_LARGA_MS - (ahora - vigentes[0])) / 1000);
    return {
      permitido: false,
      esperarSegundos: Math.max(1, espera),
      motivo: `Has alcanzado el límite de ${MAXIMO_LARGO} análisis por hora.`,
    };
  }

  vigentes.push(ahora);
  registros.set(cliente, { marcas: vigentes });
  podar(ahora);

  return { permitido: true };
}
