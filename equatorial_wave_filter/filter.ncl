; Author: Jasper (June 2024)
;         How to filter CCEW signalsa using kf_filter
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

load "kf_filter.ncl"

begin

smooth = True
folder = "/Users/xpji/spacetime_filter/filter_wave/"
folderout = "/Users/xpji/spacetime_filter/"
wave = "ig1"; input variable (mjo, kelvin, er, mrg, eig, td)

print("Reading the input file...")

f = addfile(folder + "olr.day.mean.nc", "r")
; f = addfile(folder + "mvimd.daily.regrid.1980-2019.nc", "r")

date = cd_calendar(f->time, -2)
date_start = ind(date.eq.19800101)
date_end = ind(date.eq.19991231)

latN = 15
latS = -latN

time = f->time(date_start:date_end)
lat = f->lat({latS:latN})
lon = f->lon


olr = short2flt(f->olr(time|date_start:date_end, {lat|latS:latN}, lon|:))

print("Calculating climatology and anomaly...")

TIME = cd_calendar(time, -5)
year = TIME(:, 0)
month = TIME(:, 1)
day = TIME(:, 2)
ddd = day_of_year(year, month, day)
yyyyddd = year * 1000 + ddd

if smooth then
	nhar = 3 ; number of harmonics
	climday     = clmDayTLL(olr, yyyyddd)
	climatology = smthClmDayTLL(climday,nhar)
	anomaly = calcDayAnomTLL(olr, yyyyddd, climatology)
else
	climatology = clmDayTLL(olr, yyyyddd)
	anomaly = calcDayAnomTLL(olr, yyyyddd, climatology)
end if
print("Filtering...")

filtered = new((/dimsizes(time), dimsizes(lat), dimsizes(lon)/), float, -999)

filtered!0 = "time"
filtered!1 = "lat"
filtered!2 = "lon"

filtered&time = time
filtered&lat = lat
filtered&lon = lon

filtered@units = "kg/m2s"

obsPerDay = 1

mis = -999
mis@_FillValue = -999

wave_type = wave

if (wave.eq."mjo") then
	wavenumber = (/1, 5/)
	period = (/20, 100/)
	depth = (/mis, mis/)
	filtered@long_name = "Madden-Julian Oscillations in Outgoing Longwave Radiation"
else if (wave.eq."kelvin") then
	; wavenumber = (/1, 14/)
	; period = (/2.5, 17/)
	wavenumber = (/2, 14/)
	period = (/3, 20/)
	depth = (/8, 90/)
	filtered@long_name = "Kelvin Waves in Outgoing Longwave Radiation"
else if (wave.eq."er") then
	wavenumber = (/-10, -1/)
	period = (/9, 72/)
	depth = (/8, 90/)
	filtered@long_name = "Equatorial Rossby Waves in Outgoing Longwave Radiation"
else if (wave.eq."mrg") then
	wavenumber = (/-10, -1/)
	period = (/3, 10/)
	depth = (/8, 90/)
	filtered@long_name = "Mixed Rossby-Gravity Waves in Outgoing Longwave Radiation"
else if (wave.eq."ig0") then
	wavenumber = (/1, 14/)
	period = (/1, 5/)
	depth = (/12, 50/)
	wave_type = "ig0"
	filtered@long_name = "Eastward Inertio Gravity Waves in Outgoing Longwave Radiation"
else if (wave.eq."td") then
	wavenumber = (/-20, -6/)
	; period = (/2.5, 10/)
	period = (/2.5, 5/)
	depth = (/mis, 90/)
	wave_type = "td-type"
	filtered@long_name = "Tropical Depression-Type Waves in Outgoing Longwave Radiation"
end if
end if
end if
end if
end if
end if

do y = 0, (dimsizes(lat) - 1)
	filtered(time|:, lat|y, lon|:) = (/kf_filter(anomaly(time|:, lat|y, lon|:), obsPerDay, period(0), period(1), wavenumber(0), wavenumber(1), depth(0), depth(1), wave_type)/)
	print((/lat(y)/))
end do

printVarSummary(filtered)
printMinMax(filtered, False)

print("Opening the output file...")

system("rm "+folderout+"olr." + wave + "."+latN+"_wm_2_14_smooth.nc")
g = addfile(folderout+"olr." + wave + "."+latN+"_wm_2_14_smooth.nc", "c")

print("Writing the output file...")

g->$wave$ = filtered

print("Closing the output file...")

delete(g)

end
