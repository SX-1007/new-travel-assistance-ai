/**
 * Trip limits, mirrored from the backend (`app/shared/trip_limits.py`).
 * The backend is the authority and refuses anything outside them with a
 * readable 422; these only let the form say so before a request is sent.
 * Past dates are left to the backend, which knows the time-zone slack.
 */
export const MAX_TRAVELLERS = 9;
export const MAX_TRIP_DAYS = 14;
export const MAX_CITIES = 5;
export const MAX_PLACE_NAME_LENGTH = 100;

const DAY_MS = 24 * 60 * 60 * 1000;

/** Inclusive number of days from start to end (YYYY-MM-DD strings). */
export const tripDays = (startDate: string, endDate: string): number =>
  Math.round((Date.parse(`${endDate}T00:00:00Z`) - Date.parse(`${startDate}T00:00:00Z`)) / DAY_MS) + 1;

/** The first limit the form breaks, as a message for the user, or null. */
export function tripLimitProblem(form: {
  country: string;
  cities: string[];
  numPeople: number;
  startDate: string;
  endDate: string;
}): string | null {
  if (form.country.length > MAX_PLACE_NAME_LENGTH)
    return `The destination country must be at most ${MAX_PLACE_NAME_LENGTH} characters.`;
  if (form.cities.length > MAX_CITIES)
    return `At most ${MAX_CITIES} cities can be planned in one trip.`;
  if (form.cities.some((city) => city.length > MAX_PLACE_NAME_LENGTH))
    return `Each city must be at most ${MAX_PLACE_NAME_LENGTH} characters.`;
  if (form.numPeople > MAX_TRAVELLERS)
    return `At most ${MAX_TRAVELLERS} travellers can be planned at once.`;
  const days = tripDays(form.startDate, form.endDate);
  if (days > MAX_TRIP_DAYS)
    return `A trip can be at most ${MAX_TRIP_DAYS} days long; this one is ${days} days.`;
  return null;
}
