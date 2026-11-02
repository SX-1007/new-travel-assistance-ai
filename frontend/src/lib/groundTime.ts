/**
 * groundTime.ts — the traveller's free time at the destination on each day.
 *
 * Same rule as the backend (app/modules/booking/flight_timing.py and
 * app/modules/itinerary/ground_time.py): a day's usable window is
 * 08:00-22:00 local, from 1.5 h after the outbound flight lands until 3 h
 * before the return flight leaves. Under one sight's worth of time (1.75 h)
 * makes a travel day, the only kind of day that may have no places.
 *
 * Flight times are the airports' local wall-clock times. They are handled
 * as UTC timestamps so the viewer's own time zone never shifts them.
 */
import type { DayItinerary } from './api';

const HOUR_MS = 3_600_000;
const DAY_MS = 24 * HOUR_MS;
const DAY_START_HOURS = 8;
const DAY_END_HOURS = 22;
const ARRIVAL_SETTLE_HOURS = 1.5;
const DEPARTURE_BUFFER_HOURS = 3;
const MIN_PLANNABLE_HOURS = 1.75;
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

export interface DayFreeTime {
  /** Usable hours at the destination, rounded to 0.01 h. */
  hours: number;
  /** No time for places: the traveller is in the air, or about to leave. */
  travelDay: boolean;
  /** Why the day is short, from the flight times; null on a full day. */
  note: string | null;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** An ISO date (YYYY-MM-DD) as a UTC midnight timestamp, or null. */
const dateUtc = (value: unknown): number | null => {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const [year, month, day] = value.split('-').map(Number);
  const parsed = new Date(Date.UTC(year, month - 1, day));
  return parsed.getUTCFullYear() === year && parsed.getUTCMonth() === month - 1 && parsed.getUTCDate() === day
    ? parsed.getTime()
    : null;
};

/** Local airport wall-clock time as a UTC timestamp, or null when unreadable. */
const flightTimeUtc = (flight: unknown, side: 'arrival' | 'departure'): number | null => {
  if (!isRecord(flight)) return null;
  const airport = flight[`${side}_airport`];
  const raw = flight[`${side}_time`] || (isRecord(airport) ? airport.time : undefined);
  if (typeof raw !== 'string') return null;
  const match = /^(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}):(\d{2}))?/.exec(raw.trim());
  const day = match ? dateUtc(match[1]) : null;
  if (day === null) return null;
  return day + (Number(match![2] ?? 0) * 60 + Number(match![3] ?? 0)) * 60_000;
};

const pad = (value: number) => String(value).padStart(2, '0');

const clock = (timestamp: number): string => {
  const at = new Date(timestamp);
  return `${pad(at.getUTCHours())}:${pad(at.getUTCMinutes())}`;
};

/** "16:25" on the day itself, "3 Aug at 01:00" on another day. */
const when = (timestamp: number, dayStart: number): string => {
  const sameDay = Math.floor(timestamp / DAY_MS) * DAY_MS === dayStart;
  if (sameDay) return clock(timestamp);
  const at = new Date(timestamp);
  return `on ${at.getUTCDate()} ${MONTHS[at.getUTCMonth()]} at ${clock(timestamp)}`;
};

const note = (
  day: number,
  start: number,
  end: number,
  travelDay: boolean,
  arrival: number | null,
  departure: number | null,
): string | null => {
  const landing = arrival !== null && arrival + ARRIVAL_SETTLE_HOURS * HOUR_MS > day + DAY_START_HOURS * HOUR_MS;
  const leaving = departure !== null && departure - DEPARTURE_BUFFER_HOURS * HOUR_MS < day + DAY_END_HOURS * HOUR_MS;
  if (!landing && !leaving) return null;
  const reasons = [
    landing ? `your flight lands ${when(arrival!, day)}` : null,
    leaving ? `your return flight leaves ${when(departure!, day)}` : null,
  ].filter(Boolean).join(' and ');

  if (!travelDay) {
    if (landing && leaving) return `Free ${clock(start)}–${clock(end)}, ${reasons}`;
    return landing ? `Free from ${clock(start)}, ${reasons}` : `Free until ${clock(end)}, ${reasons}`;
  }
  if (landing && arrival! >= day + DAY_MS) return `In transit, ${reasons}`;
  const airportBy = leaving ? departure! - DEPARTURE_BUFFER_HOURS * HOUR_MS : null;
  if (!landing && airportBy !== null && airportBy >= day) {
    return `Head to the airport by ${clock(airportBy)}, ${reasons}`;
  }
  return `No time at the destination, ${reasons}`;
};

/** Day number → free time, for every day with a valid date. The outbound
 * flight is day 1's first leg and the return the last day's last leg. */
export const dayFreeTimes = (itinerary: DayItinerary[]): Map<number, DayFreeTime> => {
  const result = new Map<number, DayFreeTime>();
  if (itinerary.length === 0) return result;
  const legs = (day: DayItinerary): unknown[] => (Array.isArray(day.flight) ? day.flight : []);
  const firstLegs = legs(itinerary[0]);
  const lastLegs = legs(itinerary[itinerary.length - 1]);
  const outbound = firstLegs.length > 0 ? firstLegs[0] : null;
  const returning = lastLegs.length > 0 && lastLegs[lastLegs.length - 1] !== outbound
    ? lastLegs[lastLegs.length - 1]
    : null;
  const arrival = outbound != null ? flightTimeUtc(outbound, 'arrival') : null;
  const departure = returning != null ? flightTimeUtc(returning, 'departure') : null;

  for (const day of itinerary) {
    const date = dateUtc(day.date);
    if (date === null) continue;
    let start = date + DAY_START_HOURS * HOUR_MS;
    let end = date + DAY_END_HOURS * HOUR_MS;
    if (arrival !== null) start = Math.max(start, arrival + ARRIVAL_SETTLE_HOURS * HOUR_MS);
    if (departure !== null) end = Math.min(end, departure - DEPARTURE_BUFFER_HOURS * HOUR_MS);
    const hours = Math.round(Math.max(0, (end - start) / HOUR_MS) * 100) / 100;
    const travelDay = hours < MIN_PLANNABLE_HOURS;
    result.set(day.day, { hours, travelDay, note: note(date, start, end, travelDay, arrival, departure) });
  }
  return result;
};

/** Days with no usable time at the destination. */
export const travelDays = (itinerary: DayItinerary[]): Set<number> =>
  new Set([...dayFreeTimes(itinerary)].filter(([, time]) => time.travelDay).map(([day]) => day));
