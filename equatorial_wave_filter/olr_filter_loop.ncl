; Author: Jasper (June 2024)
;         How to filter CCEW signalsa using kf_filter
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

load "/Users/xpji/spacetime_filter/kf_filter.ncl"

begin

	smooth = True


	; wave = "kelvin"; input variable (mjo, kelvin, er, mrg, eig, td)

	folder = "/Users/xpji/spacetime_filter/filter_wave/"
	folderout = "/Users/xpji/spacetime_filter/"
	wave = "kelvin"; input variable (mjo, kelvin, er, mrg, eig, td)

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

	printVarSummary(olr)

	pr=olr

	threshold = 500
	threshold_low =0
	pr@_FillValue = 1e20

	if (any(isnan_ieee(pr))) then
		replace_ieeenan (pr, pr@_FillValue, 0)
	end if

	pr = where(pr.gt.threshold, pr@_FillValue, pr)
	pr = where(pr.lt.threshold_low, pr@_FillValue, pr)

	guess     = 1

	is_cyclic = True

	nscan     = 1500

	eps       = 1.e-2

	relc      = 0.6

	opt       = 0


	poisson_grid_fill( pr, is_cyclic, guess, nscan, eps, relc, opt)

	print(all(.not.ismissing(pr)))

	printMinMax(pr,False)

	print("Calculating climatology and anomaly...")
	
	TIME = cd_calendar(time, -5)
	year = TIME(:, 0)
	month = TIME(:, 1)
	day = TIME(:, 2)
	ddd = day_of_year(year, month, day)
	yyyyddd = year * 1000 + ddd
		
	if smooth then
		nhar = 3 ; 
		climday     = clmDayTLL(olr, yyyyddd)
		climatology = smthClmDayTLL(climday,nhar)
		anomaly = calcDayAnomTLL(olr, yyyyddd, climatology)
	else
		climatology = clmDayTLL(olr, yyyyddd)
		anomaly = calcDayAnomTLL(olr, yyyyddd, climatology)
	end if
	print("Filtering...")

	waves = (/"mjo", "kelvin", "er", "mrg", "eig", "td"/)
	filtered = new((/dimsizes(time), dimsizes(lat), dimsizes(lon)/), float, -999)

	filtered!0 = "time"
	filtered!1 = "lat"
	filtered!2 = "lon"

	filtered&time = time
	filtered&lat = lat
	filtered&lon = lon

	filtered@units = "W/m^2"

	do i = 0, dimsizes(waves)-1
		
		wave_type = waves(i)
		print("Processing wave type: " + wave_type)
		if (wave.eq."mjo") then
			wavenumber = (/1, 5/)
			period = (/20, 100/)
			depth = (/mis, mis/)
			filtered@long_name = "Madden-Julian Oscillations in Outgoing Longwave Radiation"
		else if (wave.eq."kelvin") then
			wavenumber = (/2, 14/)
			period = (/3, 20/)
			depth = (/8, 90/)
			filtered@long_name = "Kelvin Waves in Precipitation"
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
		else if (wave.eq."eig") then
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

		

		obsPerDay = 1

		mis = -999
		mis@_FillValue = -999

		do y = 0, (dimsizes(lat) - 1)
			filtered(time|:, lat|y, lon|:) = (/kf_filter(anomaly(time|:, lat|y, lon|:), obsPerDay, period(0), period(1), wavenumber(0), wavenumber(1), depth(0), depth(1), wave_type)/)
			print((/lat(y)/))
		end do

		printVarSummary(filtered)
		printMinMax(filtered, False)

		print("Opening the output file...")


		g = addfile(folderout +"olr_"+ wave_type + "_"+latN+"_1980_1999.nc", "c")

		print("Writing the output file...")

		g->$wave$ = filtered

		print("Closing the output file...")

		delete(g)
		end do
end
