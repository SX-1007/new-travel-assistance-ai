import { describe, expect, it } from 'vitest';
import type { DayItinerary } from '../src/lib/api';
import { dayFreeTimes, travelDays } from '../src/lib/groundTime';

const flight = (departs: string, arrives: string) => ({
  departure_airport: { id: 'KUL', time: departs },
  arrival_airport: { id: 'BKK', time: arrives },
});

const trip = (outbound: object | null, returning: object | null, dates: string[]): DayItinerary[] =>
  dates.map((date, index) => ({
    day: index + 1,
    date,
    flight: [
      ...(index === 0 && outbound ? [outbound] : []),
      ...(index === dates.length - 1 && returning ? [returning] : []),
    ] as DayItinerary['flight'],
  }));

const notes = (itinerary: DayItinerary[]) =>
  [...dayFreeTimes(itinerary).values()].map((time) => time.note);

describe('free time per day', () => {
  it('explains the live Bangkok trip: an evening, a full day, a departure morning', () => {
    // VZ 681 KUL 15:10 → BKK 16:25; VZ 680 BKK 11:00 → KUL.
    const bangkok = trip(
      flight('2026-11-15 15:10', '2026-11-15 16:25'),
      flight('2026-11-17 11:00', '2026-11-17 14:00'),
      ['2026-11-15', '2026-11-16', '2026-11-17'],
    );

    expect(notes(bangkok)).toEqual([
      'Free from 17:55, your flight lands 16:25',
      null,
      'Head to the airport by 08:00, your return flight leaves 11:00',
    ]);
    expect([...dayFreeTimes(bangkok).values()].map((time) => time.hours)).toEqual([4.08, 14, 0]);
    expect(travelDays(bangkok)).toEqual(new Set([3]));
  });

  it('says when a day is spent in the air, with the landing date', () => {
    // Found live: KUL 19:15 → HND 01:00 two days later.
    const tokyo = trip(
      flight('2026-11-08 19:15', '2026-11-10 01:00'),
      flight('2026-11-13 19:00', '2026-11-14 03:00'),
      ['2026-11-08', '2026-11-09', '2026-11-10', '2026-11-11', '2026-11-12', '2026-11-13'],
    );

    expect(notes(tokyo)).toEqual([
      'In transit, your flight lands on 10 Nov at 01:00',
      'In transit, your flight lands on 10 Nov at 01:00',
      null,
      null,
      null,
      'Free until 16:00, your return flight leaves 19:00',
    ]);
  });

  it('names both flights on a short same-day trip', () => {
    const dayTrip = trip(
      flight('2026-11-15 06:00', '2026-11-15 08:00'),
      flight('2026-11-15 21:00', '2026-11-15 22:00'),
      ['2026-11-15'],
    );

    expect(notes(dayTrip)).toEqual([
      'Free 09:30–18:00, your flight lands 08:00 and your return flight leaves 21:00',
    ]);
  });

  it('covers a late landing and a return in the small hours', () => {
    const late = trip(
      flight('2026-11-15 18:00', '2026-11-15 21:30'),
      flight('2026-11-17 02:15', '2026-11-17 05:00'),
      ['2026-11-15', '2026-11-16', '2026-11-17'],
    );

    // Check-in for the 02:15 flight is at 23:15 the night before: day 2 is
    // still a full day, and day 3 has no time.
    expect(notes(late)).toEqual([
      'No time at the destination, your flight lands 21:30',
      null,
      'No time at the destination, your return flight leaves 02:15',
    ]);
    expect(travelDays(late)).toEqual(new Set([1, 3]));
  });

  it('cuts the evening before a return just after midnight', () => {
    const night = trip(
      flight('2026-11-15 05:00', '2026-11-15 06:00'),
      flight('2026-11-17 00:30', '2026-11-17 03:00'),
      ['2026-11-15', '2026-11-16', '2026-11-17'],
    );

    expect(notes(night)[1]).toBe('Free until 21:30, your return flight leaves on 17 Nov at 00:30');
  });

  it('gives no note when the flight times are unreadable', () => {
    const unknown = trip({ arrival_airport: { time: 'soon' } }, null, ['2026-11-15', '2026-11-16']);

    expect(notes(unknown)).toEqual([null, null]);
    expect(travelDays(unknown)).toEqual(new Set());
    expect(dayFreeTimes([]).size).toBe(0);
  });
});
