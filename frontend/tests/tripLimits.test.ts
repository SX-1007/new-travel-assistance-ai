import { describe, expect, it } from 'vitest';
import { MAX_TRIP_DAYS, tripDays, tripLimitProblem } from '../src/lib/tripLimits';

const valid = {
  country: 'Japan',
  cities: ['Tokyo'],
  numPeople: 2,
  startDate: '2026-11-01',
  endDate: '2026-11-03',
};

describe('tripLimitProblem', () => {
  it('accepts a trip just inside every limit', () => {
    expect(tripLimitProblem({
      ...valid,
      numPeople: 9,
      cities: ['A', 'B', 'C', 'D', 'x'.repeat(100)],
      endDate: '2026-11-14',
    })).toBeNull();
  });

  it('counts trip days inclusively', () => {
    expect(tripDays('2026-11-01', '2026-11-01')).toBe(1);
    expect(tripDays('2026-11-01', '2026-11-14')).toBe(MAX_TRIP_DAYS);
  });

  it('refuses each limit just outside it', () => {
    expect(tripLimitProblem({ ...valid, numPeople: 10 })).toMatch(/At most 9 travellers/);
    expect(tripLimitProblem({ ...valid, endDate: '2026-11-15' })).toMatch(/this one is 15 days/);
    expect(tripLimitProblem({ ...valid, cities: ['A', 'B', 'C', 'D', 'E', 'F'] })).toMatch(/At most 5 cities/);
    expect(tripLimitProblem({ ...valid, cities: ['x'.repeat(101)] })).toMatch(/100 characters/);
    expect(tripLimitProblem({ ...valid, country: 'x'.repeat(101) })).toMatch(/100 characters/);
  });
});
